import torch
from video_diffusion_pytorch.video_diffusion_pytorch import Unet_ecnoder, Unet, GaussianDiffusion, Trainer, cycle
from datasets.cityscape import data_load
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import random
from video_diffusion_pytorch.model_creation import create_gaussian_diffusion

def arg_parse():
    parser = argparse.ArgumentParser(description="values from bash script")
    parser.add_argument("--gpus", type=int, default=1, help="number of cuda device")
    args = parser.parse_args() 
    return args

def main(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '22490'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)  
    
    
    
    # old lunwen 
    encoder=Unet_ecnoder( dim=128, 
            dim_mults=(1,2,3,4),
            img_size=128,
            n_head_channels=128, 
            attn_resolutions=[32]
            )
  
    #new

    # encoder=Unet_ecnoder( dim=192, 
    #         dim_mults=(1,2,3,4),
    #         img_size=128,
    #         n_head_channels=192, 
    #         attn_resolutions=[32]
    #         )

    init_seed = 2222
    torch.manual_seed(init_seed)
    random.seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    data_root='/home/ysy/ysy/dataset/cityscape/leftImg8bit_sequence/leftImg8bit_sequence/hdf5'
    #test_root='/home/ysy/ysy/dataset/cityscape/leftImg8bit_sequence/leftImg8bit_sequence/forcompare/hdf5'
    train_dataloader = data_load(data_root, stage='test', batch_size=1, num_workers=0, frames_per_sample=14, distributed=False)
    #train_dataloader = data_load(data_root, stage='train', batch_size=8, num_workers=0, frames_per_sample=30, distributed=False)
    val_dataloader = data_load(data_root, stage='val', batch_size=1, num_workers=0, frames_per_sample=14, distributed=False)
  
    test_dataloader = data_load(data_root, stage='test', batch_size=1, num_workers=0, frames_per_sample=30, distributed=False)



    #test_dataloader = test_data_load(data_root='/home/ysy/ysy/dataset', image_size=256, batch_size=4, num_workers=4, distributed=False,  pin_memory=True)

    # model = Unet3D(
    #     dim = 64,
    #     dim_mults = (1, 2, 4, 8),
    # )
    
    # model = Unet( dim=128, 
    #         dim_mults=(1,2,3,4),
    #         img_size=128,
    #         n_head_channels=128, 
    #         attn_resolutions=[64]
    #         )
    
    
    # old luwne
    model = Unet( dim=128, 
            dim_mults=(1,2,3,4),
            img_size=128,
            n_head_channels=128, 
            attn_resolutions=[32]
            )
    # model = Unet( dim=192, 
    #         dim_mults=(1,2,3,4),
    #         img_size=192,
    #         n_head_channels=128, 
    #         attn_resolutions=[32,64]
    #         )

    # from torchstat import stat
    # stat(encoder,(24,128,128))

    # diffusion = GaussianDiffusion(
    #     encoder_fn = encoder,
    #     denoise_fn=model,
    #     image_size = 128,
    #     num_frames = 8,
    #     #rank=rank,
    #     timesteps = 1000,   # number of steps
    #     loss_type = 'l1'    # L1 or L2
    # ).to(rank)


    # input1=torch.randn(1,24,128,128)
    # input2=torch.randn(1)
    # input3=torch.randn(1,8,128,128)
    # input4=torch.randn(1,768,16,16)
    # from thop import profile
    # print(profile(model,inputs=(input1,input2,input3,input4)))

    diffusion = create_gaussian_diffusion( 
        steps = 1000,
        noise_schedule='linear',
        timestep_respacing='100',
        encoder=encoder,
        model=model,
        image_size = 128,
        num_frames = 8,
        rank=rank,
        timesteps = 1000,   # number of steps
        loss_type = 'l1'    # L1 or L2
    ).to(rank)


    diffusion = DDP(diffusion, device_ids=[rank])

    trainer = Trainer(
        rank = rank,
        train_dataloader = train_dataloader,
        val_dataloader = val_dataloader,
        diffusion_model=diffusion,
        train_batch_size = 1,
        train_lr = 1e-4,
        save_and_sample_every = 3000,
        train_num_steps = 1000000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = False                      # turn on mixed precision 
    )    

    #train
    trainer.pth_transfer(8)
    trainer.train()

    #trainer.inpainting(test_dataloader)
    trainer.test(test_dataloader,flag = 'prediction')


if __name__=="__main__":
    args = arg_parse()    
    mp.spawn(main, args=(args.gpus,), nprocs=args.gpus, join=True)
    #dist.barrier()
    dist.destroy_process_group()