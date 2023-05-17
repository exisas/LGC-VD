import sys
sys.path.append('/home/ysy/ysy/workspace/diffusion_models/video_diffusion/video-diffusion-data-enhancement-stm-bair/datasets')
sys.path.append('/home/ysy/ysy/workspace/diffusion_models/video_diffusion/video-diffusion-data-enhancement-stm-bair')
import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.h5 import HDF5Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import h5py

class Bair(Dataset):

    def __init__(self, data_path, frames_per_sample=5, random_time=True, random_horizontal_flip=True, color_jitter=0,
                 total_videos=-1):

        self.data_path = data_path                    # '/path/to/Datasets/Cityscapes128_h5/train' (with shard_0001.hdf5 in it)
        self.frames_per_sample = frames_per_sample
        self.random_time = random_time
        self.random_horizontal_flip = random_horizontal_flip
        self.color_jitter = color_jitter
        self.total_videos = total_videos            # If we wish to restrict total number of videos (e.g. for val)

        self.jitter = transforms.ColorJitter(hue=color_jitter)

        # Read h5 files as dataset
        self.videos_ds = HDF5Dataset(self.data_path)

       # print(f"Dataset length: {self.__len__()}")


    def __len__(self):
        return len(self.videos_ds)


    def __getitem__(self, index, time_idx=0):

        # Use `index` to select the video, and then
        # randomly choose a `frames_per_sample` window of frames in the video
        shard_idx, idx_in_shard = self.videos_ds.get_indices(index)

        data = torch.tensor([])
        flip_p = np.random.randint(2) == 0 if self.random_horizontal_flip else 0
        with h5py.File(self.videos_ds.shard_paths[shard_idx],'r') as f:
            video_len = f['len'][str(idx_in_shard)][()]
            if self.random_time and video_len > self.frames_per_sample:
                time_idx = np.random.choice(video_len - self.frames_per_sample)
            for i in range(time_idx, min(time_idx + self.frames_per_sample, video_len)):
                img = f[str(idx_in_shard)][str(i)][()]
                arr = transforms.RandomHorizontalFlip(flip_p)(transforms.ToTensor()(img))
                arr = self.jitter(arr)
                data=torch.concat((data,arr),dim=0)
  


        return data

def data_load(data_root, stage, batch_size, num_workers, frames_per_sample=14, random_time=True, random_horizontal_flip=True, color_jitter=0,
                 total_videos=-1, distributed=True,  pin_memory=True):
    data_path=os.path.join(data_root,stage)
    dataset = Bair(data_path, frames_per_sample=frames_per_sample, random_time=random_time, random_horizontal_flip=random_horizontal_flip, color_jitter=color_jitter, total_videos=total_videos)

    sampler = DistributedSampler(dataset) if distributed else None

    data_loader =  DataLoader(
            dataset,
            batch_size=batch_size,
            #shuffle=False if distributed else True,
            shuffle=True ,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=sampler
        )
    
    # prefetch_loader = PrefetchDataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         shuffle=False if distributed else True,
    #         num_workers=num_workers,
    #         pin_memory=pin_memory,
    #         sampler=sampler
    #         )

    return data_loader


if __name__=='__main__':
    data_root='/home/ysy/ysy/dataset/bair/hdf5'
   
    train_dataloader = data_load(data_root, stage='train', batch_size=1, num_workers=1, frames_per_sample=14,distributed=False)
    image = next(iter(train_dataloader))

    print(2)