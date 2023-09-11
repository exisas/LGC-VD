import torch
from model.Unet import Unet_ecnoder, Unet
from video_diffusion_pytorch.diffusion import GaussianDiffusion
from video_diffusion_pytorch.video_diffusion_pytorch import Trainer
from datasets.cityscape import data_load
import argparse
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import random
from video_diffusion_pytorch.model_creation import create_gaussian_diffusion

import yaml
from types import SimpleNamespace

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cityscape.yml", help="config path")
    args = parser.parse_args() 
    
    with open(args.config, 'r') as file:
        params = yaml.safe_load(file)
    return SimpleNamespace(**params)
    

def main(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)  


    init_seed = args.random_seed
    torch.manual_seed(init_seed)
    random.seed(init_seed)
    torch.cuda.manual_seed(init_seed)

    train_dataloader = data_load(args.data_root, stage='train', batch_size=args.batch_size, num_workers=args.dataloader_num_workers, frames_per_sample=14, distributed=False)
    val_dataloader = data_load(args.data_root, stage='val', batch_size=args.batch_size_val, num_workers=args.dataloader_num_workers, frames_per_sample=14, distributed=False)


    encoder=Unet_ecnoder( dim=args.dim, 
            dim_mults=args.dim_mults,
            img_size=args.img_size,
            n_head_channels=args.n_head_channels, 
            attn_resolutions=args.attn_resolutions
            )
    
    # #res=128
    model = Unet( dim=args.dim, 
            dim_mults=args.dim_mults,
            img_size=args.img_size,
            n_head_channels=args.n_head_channels,
            attn_resolutions=args.attn_resolutions
            )


    diffusion = create_gaussian_diffusion( 
        steps = args.total_steps, # total diffusion steps for training
        timestep_respacing=args.time_respacing, # steps for sampling
        encoder=encoder,
        model=model,
        image_size = args.img_size,
        num_frames = 8,
        rank=rank,
        loss_type = 'l1'    # L1 or L2
    ).to(rank)
    




    diffusion = DDP(diffusion, device_ids=[rank])

    trainer = Trainer(
        rank = rank,
        train_dataloader = train_dataloader,
        val_dataloader = val_dataloader,
        diffusion_model=diffusion,
        train_batch_size = 1,
        train_lr = args.train_lr,
        save_and_sample_every = args.save_steps,
        train_num_steps = args.train_steps,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = args.amp                  # turn on mixed precision 
    )    

    #train
    if args.train:
        if args.resume:
            trainer.load()
        trainer.train()
    
    else:
        trainer.pth_transfer(args.check_points)
        test_dataloader = data_load(args.data_root, stage='test', batch_size=args.batch_size_test, num_workers=args.dataloader_num_workers, frames_per_sample=30, distributed=False)
        trainer.test(test_dataloader,flag = args.task)
    #trainer.inpainting(train_dataloader)

    #inpainting
    # trainer.pth_transfer(333)
    # if rank==0:
    #      trainer.inpainting(test_dataloader)



if __name__=="__main__":
    args = arg_parse()    
    mp.spawn(main, args=(args.gpus, args), nprocs=args.gpus, join=True)
    #dist.barrier()
    dist.destroy_process_group()
