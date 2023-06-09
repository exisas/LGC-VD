import os
import glob
from numpy import append
import torch
import torchvision.transforms as transforms
import torchvision
from video_diffusion_pytorch.evaluate import calc_metric, load_i3d_pretrained, get_fvd_feats, frechet_distance
import numpy as np


psnr_perclip_list = []
ssim_perclip_list = []
lpips_perclip_list=[]
vid_perclip_list = [] 


psnr_perclip_list = []
ssim_perclip_list = []
lpips_perclip_list=[]
vid_perclip_list = [] 

psnr_best_list=[]
ssim_best_list=[]
lpips_best_list=[]

psnr_list=[]
ssim_list=[]
lpips_list=[]

video_dir = 'results/pt'

video_list = sorted(glob.glob(video_dir+r"/pre*.pt"))
video_list.sort(key = lambda x: int(x[22:-16]))

i3d = load_i3d_pretrained('cuda')

fake_feature_list=[]
ground_feature_list=[]

num_perpt=10

for i in range(len(video_list)):
    print('num:', i)

    video_tensor = torch.load(video_list[i])
    test_num=video_tensor.shape[0]//2
    ground_all=video_tensor[0:test_num]
    fake_all=video_tensor[test_num:]

    for j in range(test_num):
        fake=fake_all[j].unsqueeze(0)
        ground=ground_all[j].unsqueeze(0)
        psnr_per_clip, ssim_per_clip, lpips_per_clip  = calc_metric(ground, fake)
        psnr_list.append(psnr_per_clip)
        ssim_list.append(ssim_per_clip)
        lpips_list.append(lpips_per_clip)

        psnr_perclip_list.append(psnr_per_clip)
        ssim_perclip_list.append(ssim_per_clip)
        lpips_perclip_list.append(lpips_per_clip)

        fake_feature_list.append(get_fvd_feats(fake, i3d, fake.device, bs=10))
        ground_feature_list.append(get_fvd_feats(ground, i3d, fake.device, bs=10))
    
    if (i % num_perpt==0 ):
        psnr = max(psnr_perclip_list)
        psnr_perclip_list=[]
        psnr_best_list.append(psnr)

        ssim = max(ssim_perclip_list)
        ssim_perclip_list=[]
        ssim_best_list.append(ssim)

        lpips = min(lpips_perclip_list)
        lpips_perclip_list=[]
        lpips_best_list.append(lpips)

        f=open('result.txt','a')
        f.write( 'sample' + ': ' + str(i) + '  psnr:' + str(psnr) + '\t')
        f.write('  ssim:' + str(ssim) + '\t')
        f.write('  lpips:' + str(lpips) + '\n')
        f.close() 






avg_psnr=np.mean(psnr_best_list)
std_psnr=np.std(psnr_best_list)
avg_ssim=np.mean(ssim_best_list)
std_ssim=np.std(ssim_best_list)
avg_lpips=np.mean(lpips_best_list)
std_lpips=np.std(lpips_best_list)


fake_feature_list =  np.concatenate(fake_feature_list)
ground_feature_list =  np.concatenate(ground_feature_list)
fvd = frechet_distance(fake_feature_list,ground_feature_list)

f=open('result.txt','a')
f.write( '  avg_psnr:' + str(avg_psnr) + '\t' + 'std: ' + str(std_psnr)) 
f.write('  avg_ssim:' + str(avg_ssim) + '\t' + 'std: ' + str(std_ssim))
f.write('  avg_lpips:' + str(avg_lpips) + '\t' + 'std: ' + str(std_lpips))
f.write('  fvd:' + str(fvd) + '\n')
f.write( '  avg_psnr:' + str(np.mean(psnr_list)) + '\t') 
f.write('  avg_ssim:' + str(np.mean(ssim_list)) + '\t')
f.write('  avg_lpips:' + str(np.mean(lpips_list)) + '\t')

f.close() 




