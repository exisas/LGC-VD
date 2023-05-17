import sys
import random
import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F

from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many


#from text import tokenize, bert_embed, BERT_MODEL_DIM
#from video_diffusion_pytorch.text import tokenize, bert_embed, BERT_MODEL_DIM

import torch.distributed as dist

import functools
#from model import  layerspp


from torch.utils.tensorboard import SummaryWriter
from video_diffusion_pytorch.evaluate import calc_metric

# helpers functions

def exists(x):
    return x is not None

def noop(*args, **kwargs):
    pass

def is_odd(n):
    return (n % 2) == 1

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

# relative positional bias

class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype = torch.long, device = device)
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.module.parameters(), ma_model.module.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


        




# gaussian diffusion trainer class





# trainer class

CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# tensor of shape (channels, frames, height, width) -> gif

def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

# gif -> (channels, frame, height, width) tensor

def gif_to_tensor(path, channels = 3, transform = T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)

def identity(t, *args, **kwargs):
    return t

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5

def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))



# trainer class

class Trainer(object):
    def __init__(
        self,
        rank,
        train_dataloader,
        val_dataloader,        
        diffusion_model,
        *,
        ema_decay = 0.995,
        train_batch_size = 32,
        train_lr = 1e-4,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 10,
        results_folder = './results',
        num_sample_rows = 1,
        max_grad_norm = None
    ):
        super().__init__()
        self.device = rank
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        
        self.train_dataloader = cycle(train_dataloader),
        self.val_dataloader = cycle(val_dataloader),      

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.module.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps



        #self.ds = Dataset(folder, image_size, channels = channels, num_frames = num_frames)

        #print(f'found {len(self.ds)} videos as gif files at {folder}')
        #assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

       # self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True))
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.reset_parameters()

        self.new_frames = self.model.module.denoise_fn.new_frames
        self.cond_frames = self.model.module.denoise_fn.cond_frames
        self.total_frames = self.new_frames + self.cond_frames
        self.channels= self.model.module.denoise_fn.channels

        self.writer = SummaryWriter('')
    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.module.state_dict(),
            'ema': self.ema_model.module.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1]) for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.module.load_state_dict(data['model'], **kwargs)
        self.ema_model.module.load_state_dict(data['ema'], **kwargs)
        self.scaler.load_state_dict(data['scaler'])


   
    def train(
        self,
        prob_focus_present = 0.,
        focus_present_mask = None,
        log_fn = noop
    ):
        assert callable(log_fn)




        while self.step < self.train_num_steps:
            #for i in range(self.gradient_accumulate_every):

            all_stage_images = next(self.train_dataloader[0])
            all_stage_images = all_stage_images.to(self.device)
            p = random.random()

            for stage in range(2):
                start_idx = stage * self.channels * self.total_frames - stage * self.channels * self.cond_frames 
                end_idx = (stage+1) * self.channels * self.total_frames - stage * self.channels * self.cond_frames 
                all_images = all_stage_images[:,start_idx:end_idx,:,:].clone()
                
                b,c,h,w = all_images.shape

                if stage==0:
                    pred_x0 = torch.ones_like(all_images,device=all_images.device) * 0.5
                pred_end=pred_x0[:,-self.channels*self.cond_frames:,:,:]

                if p>0.55 :
                    flag = 'prediction'
                    pos1,pos2=[0,1]
                elif p>0.15:
                    flag = 'uncon_generation'
                    pos1=pos2=-1
                elif p>0.1:
                    flag = 'interploation'
                    pos1,pos2=[0,self.total_frames-1]
                else:
                    flag = 'random'
                    pos1,pos2=random.sample(range(-1,self.total_frames-1),2)
                    
                mask_p1=torch.ones((b,1,h,w),device=self.device)*((pos1+1)/(self.total_frames+1))
                mask_p2=torch.ones((b,1,h,w),device=self.device)*((pos2+1)/(self.total_frames+1))



                if pos1==-1:
                    cond1 = torch.zeros((b,self.channels,h,w),device = self.device)

                elif (pos1 in [0,1]) and stage==1 and flag=='prediction':
                    cond1 = pred_end[:,pos1*self.channels:(pos1+1)*self.channels,:,:].clone() 
                else:
                    cond1 = all_images[:,pos1*self.channels:(pos1+1)*self.channels,:,:].clone()


                if pos2==-1:
                    cond2 = torch.zeros((b,self.channels,h,w),device = self.device)
                elif (pos2 in [0,1]) and stage==1 and flag=='prediction':
                    cond2 = pred_end[:,pos2*self.channels:(pos2+1)*self.channels,:,:].clone() 
                else:
                    cond2 = all_images[:,pos2*self.channels:(pos2+1)*self.channels,:,:].clone() 
                cond = torch.concat((cond1,mask_p1,cond2,mask_p2),dim=1)

                with autocast(enabled = self.amp):
                    loss,pred_x0 = self.model(
                        all_images,
                        cond=cond,
                        last_clip=pred_x0,
                        prob_focus_present = prob_focus_present,
                        focus_present_mask = focus_present_mask,                       
                    )

                    self.scaler.scale(loss).backward()
                    dist.barrier()
                if self.device==0:
                    print(f'{self.step}_stage_{stage}_{pos1}_{pos2}: {loss.item()}'+'__'+ flag)
                    self.writer.add_scalar('Loss/train', loss.item(), self.step)

                log = {'loss': loss.item()}

                if exists(self.max_grad_norm):
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                    self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0 and self.device==0:
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)    

                val_all_stage_images = next(self.val_dataloader[0])
                val_all_stage_images = val_all_stage_images.to(self.device)
                p = random.random()
                for stage in range(2):
                    start_idx = stage * self.channels * self.total_frames - stage * self.channels * self.cond_frames 
                    end_idx = (stage+1) * self.channels * self.total_frames - stage * self.channels * self.cond_frames 
                    val_all_images = val_all_stage_images[:,start_idx:end_idx,:,:].clone()
                    b,c,h,w = val_all_images.shape
                    #val_all_images_list, all_masks = next(self.val_dataloader[0])
                

                    if stage==0:
                        pred_x0 = torch.ones_like(val_all_images,device=val_all_images.device) * 0.5
                    pred_end=pred_x0[:,-self.channels*self.cond_frames:,:,:]

                    if p>0.55 :
                        flag = 'prediction'
                        pos1,pos2=[0,1]
                    elif p>0.15:
                        flag = 'uncon_generation'
                        pos1=pos2=-1
                    elif p>0.1:
                        flag = 'interploation'
                        pos1,pos2=[0,self.total_frames-1]
                    else:
                        flag = 'random'
                        pos1,pos2=random.sample(range(-1,self.total_frames-1),2)

                    mask_p1=torch.ones((b,1,h,w),device=self.device)*((pos1+1)/(self.total_frames+1))
                    mask_p2=torch.ones((b,1,h,w),device=self.device)*((pos2+1)/(self.total_frames+1))

                    if pos1==-1:
                        cond1 = torch.zeros((b,self.channels,h,w),device = self.device)
                    elif (pos1 in [0,1]) and stage==1 and flag=='prediction':
                        cond1 = pred_end[:,pos1*self.channels:(pos1+1)*self.channels,:,:].clone() 
                    else:
                        cond1 = val_all_images[:,pos1*self.channels:(pos1+1)*self.channels,:,:].clone() 

                    if pos2==-1:
                        cond2 = torch.zeros((b,self.channels,h,w),device = self.device)
                    elif (pos2 in [0,1]) and stage==1 and flag=='prediction':
                        cond2 = pred_end[:,pos2*self.channels:(pos2+1)*self.channels,:,:].clone() 
                    else:
                        cond2 = val_all_images[:,pos2*self.channels:(pos2+1)*self.channels,:,:].clone() 
                    cond = torch.concat((cond1,mask_p1,cond2,mask_p2),dim=1)


                    val_cond = normalize_img(cond)
                    

                    if p>0.55:
                        if stage==0 :
                            pred_x0 = normalize_img(pred_x0)
                            cond_feature = self.model.module.get_cond_feature(pred_x0)
                            sample_img_stage1 = self.ema_model.module.sample(val_cond, cond_feature)
                            pred_x0=sample_img_stage1.detach()
                        else:
                            pred_x0 = normalize_img(pred_x0)
                            cond_feature = self.model.module.get_cond_feature(pred_x0)


                            sample_img_stage2 = self.ema_model.module.sample(val_cond,cond_feature)                           
                            sample_img = torch.concat((sample_img_stage1,sample_img_stage2[:,self.channels*self.cond_frames:,:,:]),dim=1)
                            sample_img = reshape5D(sample_img)
                            val_all_stage_images = reshape5D(val_all_stage_images)

                            save_img = torch.concat((val_all_stage_images,sample_img),dim=0)
                            one_gif = rearrange(save_img, '(i j) c f h w -> c f (i h) (j w)', i = self.num_sample_rows*2)

                            video_path = str(self.results_folder / str(f'{milestone}_prediction.gif'))
                            video_tensor_to_gif(one_gif, video_path)
                            
                            
                    else:
                        if stage==0:
                            pass
                        else:
                            pred_x0 = normalize_img(pred_x0)
                            cond_feature = self.model.module.get_cond_feature(pred_x0)
                            sample_img=self.ema_model.module.sample(val_cond,cond_feature)
                            sample_img = reshape5D(sample_img)
                            val_all_images = reshape5D(val_all_images)
                            save_img = torch.concat((val_all_images, sample_img),dim=0)
                            one_gif = rearrange(save_img, '(i j) c f h w -> c f (i h) (j w)', i = self.num_sample_rows*2)
                            video_path = str(self.results_folder / str(f'{milestone}_{pos1}_{pos2}.gif'))
                            video_tensor_to_gif(one_gif, video_path)






            dist.barrier()

            self.step += 1

        print('training completed')

        
    
    def inpainting(self, dataloader):
        self.ema_model.eval()
        psnr_list = []
        ssim_list = []
        lpips_list=[]
        vid_list = []
        
        channel = 3
        cond_frame = self.cond_frames
        new_frame = self.new_frames 
        total_frame = cond_frame + new_frame

        has_cond = True # For inpainting without cond, unconditional generation
        #has_cond = True # For inpainting witho cond, prediction
        #1525
        for nums, data in enumerate(dataloader):
            if nums==256:
                break
            psnr_perclip_list = []
            ssim_perclip_list = []
            lpips_perclip_list=[]
            vid_perclip_list = []
            
            all_images= data
            all_images = all_images.to(self.device)


            all_masks = torch.zeros_like(all_images,device=self.device)
            
            num_every_sample=10
            all_images=all_images.repeat((num_every_sample,1,1,1))
            all_masks=all_masks.repeat((num_every_sample,1,1,1))
            for sample_num in range(1):
                ground_all_images = all_images.clone()
                all_images = reshape5D(all_images)


                
                num_step = self.model.module.num_timesteps
                    
                total_clips = (ground_all_images.shape[1] // channel - cond_frame) // new_frame
                res = (ground_all_images.shape[1] // channel - cond_frame) % new_frame
                if res:
                    total_clips += 1
                print('total_clips:',total_clips)
                for num_clips in range(total_clips):
                    if res:
                        if num_clips < total_clips -1 :
                            ground_clip = ground_all_images[:,(num_clips * new_frame * channel) :(num_clips * new_frame * channel) + channel * total_frame,:,:].clone()
                            b,cf,h,w, = ground_clip.shape
                            masks = all_masks[:,(num_clips * new_frame * channel):(num_clips * new_frame * channel) + channel*total_frame,:,:].clone()
                            if (num_clips==0) and (has_cond==False):
                                pass
                            else:
                                masks[:,0:channel * cond_frame,:,:]=1
                            
                        else:
                            ground_clip = ground_all_images[:,- channel * total_frame :,:,:].clone()
                            b,cf,h,w, = ground_clip.shape
                            masks= all_masks[:,- channel * total_frame :,:,:].clone()
                            masks[:,0:-channel * res,:,:] = 1
                            
                    else:
                        ground_clip = ground_all_images[:,(num_clips * new_frame * channel) :(num_clips * new_frame * channel) + channel * total_frame,:,:].clone()
                        masks = all_masks[:,(num_clips * new_frame * channel):(num_clips * new_frame * channel) + channel*total_frame,:,:].clone()
                        masks[:,0:channel * cond_frame,:,:]=1
                        b,cf,h,w, = ground_clip.shape
                        if (num_clips==0) and (has_cond==False):
                            pass
                        else:
                            masks[:,0:channel * cond_frame,:,:]=1

                    # with cond feature
                    if num_clips==0:
                        pred_x0 = torch.ones_like(ground_clip,device=ground_clip.device) * 0.5
                    else:
                        pred_x0 = ground_all_images[:,((num_clips-1) * new_frame * channel) :((num_clips-1) * new_frame * channel) + channel * total_frame,:,:].clone()
                
                    #without cond feature
                    #pred_x0 = torch.ones_like(ground_clip,device=ground_clip.device) * 0.5

                    pred_x0 = normalize_img(pred_x0)

                    cond_feature = self.model.module.get_cond_feature(pred_x0)
                    # if num_clips==0:   
                    #     p = random.random()
                    #     if p>0.5:
                    #         pos1,pos2=[0,1]
                    #         have_cond=2
                    #     else :
                    #         pos1=pos2=-1
                    #         have_cond=0
                    # else:


                    if (num_clips==0) and (has_cond==False):
                        pos1=pos2=-1
                    else:
                        pos1,pos2=[0,1]
                    
                    
                    mask_p1=torch.ones((b,1,h,w),device=self.device)*((pos1+1)/(self.total_frames+1))
                    mask_p2=torch.ones((b,1,h,w),device=self.device)*((pos2+1)/(self.total_frames+1))
                    if pos1==-1:
                        cond1 = torch.zeros((b,self.channels,h,w),device = self.device)
                    else:
                        cond1 = ground_clip[:,pos1*self.channels:(pos1+1)*self.channels,:,:].clone() 
                    if pos2==-1:
                        cond2 = torch.zeros((b,self.channels,h,w),device = self.device)
                    else:
                        cond2 = ground_clip[:,pos2*self.channels:(pos2+1)*self.channels,:,:].clone() 
                    cond = torch.concat((cond1,mask_p1,cond2,mask_p2),dim=1)

                    cond_clip = normalize_img(cond)

                    ground_clip = normalize_img(ground_clip)       
                    #cond_clip = ground_clip[:,:cond_frame*channel,:,:]
                    clips = ground_clip.clone()
                    #masks = torch.ones_like(ground_clip, device=self.device)
                    #b,cf,h,w, = ground_clip.shape
                    # mask_p = torch.ones((b,channel*cond_frame,h,w),device= self.device)
                    # masks = torch.concat((mask_p,masks),dim=1)
                    #masks[:,:,3:,74:162,74:162]=0
                    masked_clip = clips * masks

                    b, *_ = ground_clip.shape
                    noise = torch.randn_like(masked_clip, device=self.device)
                    #num_step = 700 ,800
                    #num_step = 700
                    

                    #first_frame = first_frame.unsqueeze(2)
                    #first_frames = first_frame.expand(ground_clip.shape)
                    ts =  get_schedule_jump(num_step,1,1,1)

                    skip_step= 0
                    
                    #t = torch.tensor(ts[skip_step]).expand(b).to(self.device)
                    #img = self.ema_model.module.q_sample(x_start=first_frames, t=t, noise=noise)
                    img = torch.randn(masked_clip.shape, device=self.device)

                    for i in tqdm(range(skip_step, len(ts)-1), desc='sampling loop time step', total=len(ts)-1-skip_step):
                        if ts[i+1] < ts[i]:
                            t = torch.tensor(ts[i]).expand(b).to(self.device)
                            img = self.ema_model.module.p_sample(img, torch.full((b,), ts[i], device=self.device, dtype=torch.long), cond = cond_clip, cond_feature=cond_feature, cond_scale = 1.)
                            #img = self.ema_model.module.ddim_sample(img, torch.full((b,), ts[i], device=self.device, dtype=torch.long), cond = cond_clip, cond_scale = 1.)
                            ground_part = self.ema_model.module.q_sample(x_start=masked_clip, t=t, noise=noise)
                            img = ground_part * masks + img * (1 - masks)
                        else:
                            t = torch.tensor(ts[i]).expand(b).to(self.device)
                            #img = self.ema_model.module.q_diffuse_one_step(img,t)
                            img[:,cond_frame*channel:,:,:] = self.ema_model.module.q_diffuse_one_step(img[:,cond_frame*channel:,:,:],t)


                    img = masked_clip * masks + img * (1 - masks)
                    
                    img = unnormalize_img(img)

                    if res:
                        if num_clips < total_clips -1 :
                            ground_all_images[:,(num_clips * new_frame * channel) + channel * cond_frame:(num_clips * new_frame * channel) + channel * total_frame,:,:] = img[:,channel * cond_frame:,:,:]             
                        else:
                            ground_all_images[:,- channel * res :,:,:] = img[:,channel * res:,:,:] 
                        
                    else:
                        ground_all_images[:,(num_clips * new_frame * channel) + channel * cond_frame:(num_clips * new_frame * channel) + channel * total_frame,:,:] = img[:,channel * cond_frame:,:,:]    

                blue=torch.zeros_like(all_masks,device=self.device)
                blue[:,2:90:3,:,:]=0.2
                all_masked_images = reshape5D(ground_all_images + blue*(1-all_masks))
                ground_all_images = reshape5D(ground_all_images)
                
                save_imgs = torch.concat((all_images,all_masked_images,ground_all_images),dim=0)
                one_gif = rearrange(F.pad(save_imgs,(2, 2, 2, 2)), '(i j) c f h w -> c f (i h) (j w)', i = self.num_sample_rows*3)
                video_path = str(self.results_folder  /str(f'prediction_{nums}_sample_num_{sample_num}.gif'))
                #video_path = str(self.results_folder / 'without_feature' /str(f'prediction_{nums}_sample_num_{sample_num}.gif'))
                video_tensor_to_gif(one_gif, video_path)

                save_tensor = torch.concat((all_images,ground_all_images),dim=0)
                #torch.save(save_tensor,f'/home/ysy/ysy/workspace/diffusion_models/video_diffusion/video-diffusion-data-enhancement-stm/results/pt/without_feature/prediction_{nums}_sample_num_{sample_num}.pt')
                torch.save(save_tensor,f'/home/ysy/ysy/workspace/diffusion_models/video_diffusion/video-diffusion-data-enhancement-stm/results/pt/prediction_{nums}_sample_num_{sample_num}.pt')
                #torchvision.io.write_video(f'results/video_{nums}_ts111_sample_num_{sample_num}.mp4', 255 * torch.concat((all_images[0,...],ground_all_images[0,...]),dim=2).permute(1,2,3,0).cpu(),fps=12)
                

                #psnr_per_clip, ssim_per_clip, lpips_per_clip ,vid_per_clip = calc_metric(all_images, ground_all_images)
                

                all_images=reshape4D(all_images)


        #         f=open('result.txt','a')
        #         #f.write('step: ' + str(self.model.module.num_timesteps) + '\n')
        #         f.write(str(nums) +'_' + str(sample_num) + '  psnr:' + str(psnr_per_clip) + '\t')
        #         f.write('  ssim:' + str(ssim_per_clip) + '\t')
        #         f.write('  lpips:' + str(lpips_per_clip) + '\t')
        #         f.write('  vid:' + str(vid_per_clip) + '\n')
        #         f.close() 
        #         psnr_perclip_list.append(psnr_per_clip)
        #         ssim_perclip_list.append(ssim_per_clip)
        #         lpips_perclip_list.append(lpips_per_clip)
        #         vid_perclip_list.append(vid_per_clip)
            
        #     psnr_clipavg=sum(psnr_perclip_list)/len(psnr_perclip_list)
        #     ssim_clipavg=sum(ssim_perclip_list)/len(ssim_perclip_list)
        #     lpips_clipavg=sum(lpips_perclip_list)/len(lpips_perclip_list)
        #     vid_clipavg=sum(vid_perclip_list)/len(vid_perclip_list)

        #     psnr_list.append(psnr_clipavg)
        #     ssim_list.append(ssim_clipavg)
        #     lpips_list.append(lpips_clipavg)
        #     vid_list.append(vid_clipavg)

        #     avg_psnr = sum(psnr_list) / len(psnr_list)
        #     avg_ssim = sum(ssim_list) / len(ssim_list)
        #     avg_lpips = sum(lpips_list) / len(lpips_list)
        #     avg_vid = sum(vid_list) / len(vid_list)

        # f=open('result.txt','a')
        # #f.write('step: ' + str(self.model.module.num_timesteps) + '\n')
        # f.write(str(nums) + '  psnr:' + str(psnr_clipavg) + '\t')
        # f.write('  avg_ssim:' + str(ssim_clipavg) + '\t')
        # f.write('  avg_lpips:' + str(lpips_clipavg) + '\t')
        # f.write('  avg_vid:' + str(vid_clipavg) + '\t')
        # f.write('  avg_all_psnr:' + str(avg_psnr) + '\t')
        # f.write('  avg_all_ssim:' + str(avg_ssim) + '\t')
        # f.write('  avg_all_lpips:' + str(avg_lpips) + '\t')
        # f.write('  avg_all_vid:' + str(avg_vid) + '\n')

        # f.close() 


    def test(self, dataloader, flag):
        self.ema_model.eval()
        
        channel = 3
        cond_frame = self.cond_frames
        new_frame = self.new_frames 
        total_frame = cond_frame + new_frame    

        for nums, data in enumerate(dataloader):
            # if nums < 64:
            #     continue
            if nums==256:
                break
            print('num:',nums)
            all_images = data
            #torch.save(all_images,f'cityscape_test/{nums}.pt')
            #continue
            all_images = all_images.to(self.device)
            
            num_every_sample=10
            all_images=all_images.repeat((num_every_sample,1,1,1))
            for sample_num in range(10):
                ground_all_images = all_images.clone()
                all_images = reshape5D(all_images)

                num_step = self.model.module.num_timesteps
                    
                total_clips = (ground_all_images.shape[1] // channel - cond_frame) // new_frame
                res = (ground_all_images.shape[1] // channel - cond_frame) % new_frame
                if res:
                    total_clips += 1
                print('total_clips:',total_clips)
                for num_clips in range(total_clips):
                    
                    if res:
                        if num_clips < total_clips -1 :
                            ground_clip = ground_all_images[:,(num_clips * new_frame * channel) :(num_clips * new_frame * channel) + channel * total_frame,:,:].clone()                         
                        else:
                            ground_clip = ground_all_images[:,- channel * total_frame :,:,:].clone()
                            
                    else:
                        ground_clip = ground_all_images[:,(num_clips * new_frame * channel) :(num_clips * new_frame * channel) + channel * total_frame,:,:].clone()    
                    
                    b,cf,h,w, = ground_clip.shape

                    # with cond feature
                    if num_clips==0:
                       pred_x0 = torch.ones_like(ground_clip,device=ground_clip.device) * 0.5
                    else:
                       pred_x0 = img.clone()
                
                    #without cond feature
                    #pred_x0 = torch.ones_like(ground_clip,device=ground_clip.device) * 0.5

                    pred_x0 = normalize_img(pred_x0)

                    if num_clips==0:
                        cond_feature = self.model.module.get_cond_feature(pred_x0)
                    else :
                        cond_feature = self.model.module.get_cond_feature(pred_x0)
                        cond_feature = (cond_feature + last_feature)/2
                    
                    last_feature = cond_feature

                    if num_clips==0:
                        if flag=='live_photo':
                            pos1,pos2=[0,-1]
                        if flag=='prediction':
                            pos1,pos2=[0,1]
                        if flag=='generation':
                            pos1,pos2=[-1,-1]
                    else:
                        pos1,pos2=[0,1]
                    
                    
                    mask_p1=torch.ones((b,1,h,w),device=self.device)*((pos1+1)/(self.total_frames+1))
                    mask_p2=torch.ones((b,1,h,w),device=self.device)*((pos2+1)/(self.total_frames+1))
                    if pos1==-1:
                        cond1 = torch.zeros((b,self.channels,h,w),device = self.device)
                    else:
                        cond1 = ground_clip[:,pos1*self.channels:(pos1+1)*self.channels,:,:].clone() 
                    if pos2==-1:
                        cond2 = torch.zeros((b,self.channels,h,w),device = self.device)
                    else:
                        cond2 = ground_clip[:,pos2*self.channels:(pos2+1)*self.channels,:,:].clone() 
                    cond = torch.concat((cond1,mask_p1,cond2,mask_p2),dim=1)

                    cond_clip = normalize_img(cond)

                    clips = normalize_img(ground_clip).clone()      
                                                          
                    img = torch.randn(clips.shape, device=self.device)
                    
                    for i in tqdm(reversed(range(0, num_step)), desc='sampling loop time step', total=num_step):
                        img = self.ema_model.module.p_sample(img, torch.full((b,), i, device=self.device, dtype=torch.long), cond = cond_clip, cond_feature=cond_feature, cond_scale = 1.)
                    
                    img = unnormalize_img(img)

                    if res:
                        if num_clips < total_clips -1 :
                            ground_all_images[:,(num_clips * new_frame * channel) :(num_clips * new_frame * channel) + channel * total_frame,:,:] = img
                        else:
                            ground_all_images[:,- channel * res :,:,:] = img[:,-channel * res:,:,:] 
                            #ground_all_images[:,- channel * res :,:,:] = img[:, channel * cond_frame : channel*(res+cond_frame),:,:] 
                        
                    else:
                        ground_all_images[:,(num_clips * new_frame * channel) :(num_clips * new_frame * channel) + channel * total_frame,:,:] = img

                
                ground_all_images = reshape5D(ground_all_images)
                
                save_imgs = torch.concat((all_images, ground_all_images),dim=0)
                one_gif = rearrange(F.pad(save_imgs,(2, 2, 2, 2)), '(i j) c f h w -> c f (i h) (j w)', i = self.num_sample_rows*2)
                video_path = str(self.results_folder  /str(f'{flag}_{nums}_sample_num_{sample_num}.gif'))
                video_tensor_to_gif(one_gif, video_path)

                save_tensor = torch.concat((all_images,ground_all_images),dim=0)
                torch.save(save_tensor,f'results/pt/{flag}_{nums}_sample_num_{sample_num}.pt')

                all_images=reshape4D(all_images)




    def pth_transfer(self, milestone):
        if milestone == -1:
            all_milestones = [int(p.stem.split('-')[-1]) for p in Path(self.results_folder).glob('**/*.pt')]
            assert len(all_milestones) > 0, 'need to have at least one milestone to load from latest checkpoint (milestone == -1)'
            milestone = max(all_milestones)
        
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))
        self.scaler.load_state_dict(data['scaler'])
        self.step = data['step']

        model_dict = self.model.state_dict()
        model_dict = self.model.module.state_dict()
        state_dict = {k:v for k,v in data['model'].items() if (k in model_dict.keys() and ('betas' != k) and ('alphas_cumprod' != k)  and ('alphas_cumprod_prev' != k) and ('sqrt_one_minus_alphas_cumprod' != k)  and ('log_one_minus_alphas_cumprod' != k)  and ('sqrt_recip_alphas_cumprod' != k)  and ('sqrt_recipm1_alphas_cumprod' != k)  and ('posterior_variance' != model_dict.keys())  and ('posterior_log_variance_clipped' != k)  and ('posterior_mean_coef1' != k)  and ('posterior_mean_coef2' != k) and ('sqrt_alphas_cumprod' != k)  and ('posterior_variance' != k) )}
        model_dict.update(state_dict)
        self.model.module.load_state_dict(model_dict)

        model_dict = self.ema_model.module.state_dict()
        state_dict = {k:v for k,v in data['ema'].items() if (k in model_dict.keys() and ('betas' != k) and ('alphas_cumprod' != k)  and ('alphas_cumprod_prev' != k) and ('sqrt_one_minus_alphas_cumprod' != k)  and ('log_one_minus_alphas_cumprod' != k)  and ('sqrt_recip_alphas_cumprod' != k)  and ('sqrt_recipm1_alphas_cumprod' != k)  and ('posterior_variance' != model_dict.keys())  and ('posterior_log_variance_clipped' != k)  and ('posterior_mean_coef1' != k)  and ('posterior_mean_coef2' != k) and ('sqrt_alphas_cumprod' != k)  and ('posterior_variance' != k) )}            
        model_dict.update(state_dict)
        self.ema_model.module.load_state_dict(model_dict)
    
      

    def fast_sample(self, dataloader):
        dataloader = cycle(dataloader)
        all_images, all_masks = next(dataloader)
        all_images = all_images.to(self.device)
        all_masks = torch.ones_like(all_images, device=self.device)
        all_masks[:,:,2:,74:162,74:162]=0
        B = all_images.shape[0]
        # all_images_1 = all_images.reshape(B,-1,self.image_size,self.image_size)
        # imgs = all_images_1[:,2*3:7*3,:,:]
        # cond = all_images_1[:,0:2*3,:,:]

        
        one_gif = rearrange(F.pad(all_images, (2, 2, 2, 2)), '(i j) c f h w -> c f (i h) (j w)', i = self.num_sample_rows)
        video_path = str(self.results_folder / str(f'ground_sample.gif'))
        video_tensor_to_gif(one_gif, video_path)
        



        for num_step in [1000]:
            ground_all_images = all_images
            total_clips = (all_images.shape[2]-2) // 5 
            
            for num_clips in range(total_clips):
                ground_clip = ground_all_images[:,:,(num_clips * 5) :((num_clips + 1) * 5 + 2),:,:]
                ground_clip = normalize_img(ground_clip)
                b, c, f, h, w = ground_clip.shape
                clips = ground_clip[:,:,2:7,:,:]
                clips_conds = ground_clip[:,:,0:2,:,:]
                
                clips_concat = torch.tensor([],device=clips.device)
                for i in range(clips.shape[2]):
                    clips_concat = torch.concat((clips_concat, clips[:,:,i,:,:]),dim=1)

                clips_conds_concat = torch.tensor([],device=clips_conds.device)
                for i in range(clips_conds.shape[2]):
                    clips_conds_concat = torch.concat((clips_conds_concat, clips_conds[:,:,i,:,:]),dim=1)



                noise = torch.randn_like(clips_concat, device=self.device)

                img = torch.randn(clips_concat.shape, device=self.device)
                for i in tqdm(reversed(range(0, num_step)), desc='sampling loop time step', total=num_step):
                    img = self.ema_model.module.p_sample(img, torch.full((b,), i, device=self.device, dtype=torch.long), cond = clips_conds_concat, cond_scale = 1.)
                    #ground_part = self.ema_model.module.q_sample(x_start=masked_clips, t=t, noise=noise)
                    #img = ground_part * masks + img * (1 - masks)
                   
                
                img = unnormalize_img(img)

                img1 = []
                for j in range(0,img.shape[1] // 3 ):
                    img1.append(img[:,j*3:(j+1)*3,:,:])
                img1 = torch.stack(img1).permute(1,2,0,3,4)

                ground_all_images[:,:,(num_clips * 5 + 2 ) :((num_clips + 1) * 5 + 2),:,:] = img1

            one_gif = rearrange(ground_all_images, '(i j) c f h w -> c f (i h) (j w)', i = self.num_sample_rows)
            video_path = str(self.results_folder / str(f'fast_sample{num_step}.gif'))
            video_tensor_to_gif(one_gif, video_path)
        



def reshape5D(img):
    img1 = []
    for j in range(0,img.shape[1] // 3 ):
        img1.append(img[:,j*3:(j+1)*3,:,:])
    img1 = torch.stack(img1).permute(1,2,0,3,4)
    return img1

def reshape4D(img):
    img1 = torch.tensor([],device=img.device)
    for j in range(0,img.shape[2]):
        img1=torch.concat((img1,img[:,:,j,:,:]),dim=1)
    return img1

def save_video(video,stage):
    import torchvision
    for i in range(video.shape[1]//3):
        torchvision.utils.save_image(video[0,i*3:(i+1)*3,:,:],f'test_graph/{stage}_img_{i}.jpg')