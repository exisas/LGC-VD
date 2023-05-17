import torch
from torch import nn, einsum
import functools
from model import utils
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb



ResnetBlockDDPM = utils.ResnetBlockDDPMpp

class Unet_ecnoder(nn.Module):
    def __init__(
        self,
        dim,
        dim_mults=(1,2,2,2),
        channels = 3,
        init_dim = None,
        num_frames = 8,
        img_size = 256,
        attn_resolutions = [8, 16, 32],
        n_head_channels = 64,
    ):
        super().__init__()
        self.channels = channels

        # MCVD configuration
        self.num_frames = num_frames
        
        self.is3d = False
        self.pseudo3d = False
        self.img_size = img_size
        self.ch_mult = dim_mults
        self.attn_resolutions = attn_resolutions 
        dropout = 0.0
        self.num_resolutions = len(self.ch_mult)
        self.all_resolutions =  [self.img_size // (2 ** i) for i in range(self.num_resolutions)]
       


        # initial conv
        #self.init_conv = nn.Conv2d(channels*self.num_frames, dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.init_conv = nn.Conv2d(channels*num_frames, dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        # dimensions
        init_dim=dim
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))


        init_scale = 0
        skip_rescale = True

        act = nn.SiLU()
        
    
        
        
        AttnBlock = functools.partial(utils.AttnBlockpp,
                                                      init_scale=init_scale,
                                                      skip_rescale=skip_rescale, n_head_channels= n_head_channels)


        ResnetBlock = functools.partial(ResnetBlockDDPM,
                                          act=act,
                                          dropout=dropout,
                                          init_scale=init_scale,
                                          skip_rescale=skip_rescale,
                                          n_frames = self.num_frames,
                                          conv_shortcut=True,
                                          act3d=True) # Activation here as per https://arxiv.org/abs/1809.04096


        # layers
        self.downs = nn.ModuleList([])
        self.mids = nn.ModuleList([])
        num_resolutions = len(in_out)

        # modules for all layers
        # downs
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(nn.ModuleList([
                ResnetBlock(in_ch=dim_in, out_ch=dim_out),
                AttnBlock(channels=dim_out) if self.all_resolutions[ind] in self.attn_resolutions else nn.Identity(),
                ResnetBlock(in_ch=dim_out, out_ch=dim_out),
                AttnBlock(channels=dim_out) if self.all_resolutions[ind] in self.attn_resolutions else nn.Identity(),
                nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=2, padding=1) if (ind != num_resolutions - 1) else nn.Identity()
                
            ]))

        # middle
        mid_dim = dims[-1]
        self.midblock1 = ResnetBlock(in_ch=mid_dim)
        self.midblock2 = AttnBlock(channels=mid_dim)
        self.midblock3 = ResnetBlock(in_ch=mid_dim)


        self.norm = utils.get_act_norm(act=nn.SiLU(), act_emb=nn.SiLU(), norm='none', ch=dim_in, is3d=self.is3d, n_frames=self.num_frames)
        


    def forward(
        self,
        x
    ):
        #assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'

        # x   B CF H W

        x = self.init_conv(x)
        
        h=[]

        for resblock1, attnblock1, resblock2, attnblock2, blockdown in self.downs:

            x = resblock1(x)
            x = attnblock1(x)
            x = resblock2(x)
            x = attnblock2(x)
            h.append(x)
            x = blockdown(x)

        x = self.midblock1(x)
        x = self.midblock2(x)
        x = self.midblock3(x)

        return x



class Unet(nn.Module):
    def __init__(
        self,
        dim,
        dim_mults=(1,2,2,2),
        channels = 3,
        init_dim = None,
        cond_frames = 2,
        new_frames = 6,
        img_size = 256,
        num_res_blocks = 2,
        attn_resolutions = [8, 16, 32],
        n_head_channels = 64,
    ):
        super().__init__()
        self.channels = channels

        #configuration
        self.cond_frames = cond_frames
        self.new_frames = new_frames
        self.num_frames = cond_frames + new_frames
        
        self.is3d = False
        self.pseudo3d = False
        self.img_size = img_size
        self.ch_mult = dim_mults
        self.num_res_blocks =  num_res_blocks 
        self.attn_resolutions = attn_resolutions 
        dropout = 0.0
        self.num_resolutions = len(self.ch_mult)
        self.all_resolutions =  [self.img_size // (2 ** i) for i in range(self.num_resolutions)]
       
        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # initial conv
        #self.init_conv = nn.Conv2d(channels*self.num_frames, dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.init_conv = nn.Conv2d(channels*self.num_frames+(channels+1)*self.cond_frames, dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        # dimensions
        init_dim=dim
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))


        init_scale = 0
        skip_rescale = True

        act = nn.SiLU()
        
        AttnBlockDown = AttnBlockUp = functools.partial(utils.AttnBlockpp,
                                                      init_scale=init_scale,
                                                      skip_rescale=skip_rescale, n_head_channels= n_head_channels)

        cross_atten = functools.partial(utils.AttnBlock_cross,
                                                      init_scale=init_scale,
                                                      skip_rescale=skip_rescale, n_head_channels= n_head_channels)

        
        


        ResnetBlockDown = functools.partial(ResnetBlockDDPM,
                                          act=act,
                                          dropout=dropout,
                                          init_scale=init_scale,
                                          skip_rescale=skip_rescale,
                                          temb_dim=time_dim,
                                          n_frames = self.num_frames,
                                          conv_shortcut=True,
                                          act3d=True) # Activation here as per https://arxiv.org/abs/1809.04096
        ResnetBlockUp = functools.partial(ResnetBlockDDPM,
                                    act=act,
                                    dropout=dropout,
                                    init_scale=init_scale,
                                    skip_rescale=skip_rescale,
                                    temb_dim=time_dim,
                                    n_frames = self.num_frames,
                                    conv_shortcut=True,
                                    act3d=True) # Activation here as per https://arxiv.org/abs/1809.04096

        # layers
        self.downs = nn.ModuleList([])
        self.mids = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # modules for all layers
        # downs
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(nn.ModuleList([
                ResnetBlockDown(in_ch=dim_in, out_ch=dim_out),
                AttnBlockDown(channels=dim_out) if self.all_resolutions[ind] in self.attn_resolutions else nn.Identity(),
                ResnetBlockDown(in_ch=dim_out, out_ch=dim_out),
                AttnBlockDown(channels=dim_out) if self.all_resolutions[ind] in self.attn_resolutions else nn.Identity(),
                nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=2, padding=1) if (ind != num_resolutions - 1) else nn.Identity()
                
            ]))

        # middle
        mid_dim = dims[-1]
        self.midblock1 = ResnetBlockDown(in_ch=mid_dim)
        self.midblock2 = AttnBlockDown(channels=mid_dim)
        self.midblock3 = ResnetBlockUp(in_ch=mid_dim)
        self.midblock4 = cross_atten(channels=mid_dim)


        # 128
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            self.ups.append(nn.ModuleList([
                ResnetBlockUp(in_ch=dim_out * 2, out_ch=dim_in) ,    
                AttnBlockUp(channels=dim_in) if self.all_resolutions[ind] in self.attn_resolutions else nn.Identity(),
                ResnetBlockUp(in_ch=dim_in, out_ch=dim_in), 
                AttnBlockUp(channels=dim_in) if self.all_resolutions[ind] in self.attn_resolutions else nn.Identity(),
                ResnetBlockUp(in_ch=dim_in, out_ch=dim_in),    
                nn.ConvTranspose2d(dim_in, dim_in, kernel_size=3, stride=2, padding=1, output_padding=1) if (ind != num_resolutions - 1) else nn.Identity()
            ]))

        # finial conv

        self.norm = utils.get_act_norm(act=nn.SiLU(), act_emb=nn.SiLU(), norm='none', ch=dim_in, is3d=self.is3d, n_frames=self.num_frames)
        self.finial_conv = nn.Conv2d(dim_in, channels*self.num_frames, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        


    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob = 0., **kwargs)
        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, null_cond_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond = None,
        cond_feature = None,
        null_cond_prob = 0.,
        focus_present_mask = None,
        prob_focus_present = 0.  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        #assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'

        # x   B CF H W
       
        x = torch.concat((cond,x),dim=1)

        t = self.time_mlp(time) 
        
        x = self.init_conv(x)
        
        h=[]

        for resblock1, attnblock1, resblock2, attnblock2, blockdown in self.downs:

            x = resblock1(x,t)
            x = attnblock1(x)
            x = resblock2(x,t)
            x = attnblock2(x)
            h.append(x)
            x = blockdown(x)

        x = self.midblock1(x,t)
        x = self.midblock2(x)
        x = self.midblock3(x,t)
        x = self.midblock4(x, cond_feature)
        

        for resblock1, attnblock1, resblock2, attnblock2, resblock3, blockup in self.ups:

            x = torch.cat((x, h.pop()), dim = 1)
            x = resblock1(x,t)
            x = attnblock1(x)
            x = resblock2(x,t)
            x = attnblock2(x)
            x = resblock3(x,t)
            x = blockup(x)


        x = self.norm(x)
        x = self.finial_conv(x)
        

        return x