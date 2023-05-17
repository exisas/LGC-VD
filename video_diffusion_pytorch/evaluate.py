import torch
import os
import math
import os.path as osp
import math
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips


# https://github.com/universome/fvd-comparison
i3D_WEIGHTS_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt"

def load_i3d_pretrained(device=torch.device('cpu')):
    filepath = '/home/ysy/ysy/workspace/diffusion_models/video_diffusion/video-diffusion-pytorch-main-2d-channel-atten-cond/check_points/i3d_torchscript.pt'
    if not os.path.exists(filepath):
        os.system(f"wget {i3D_WEIGHTS_URL} -O {filepath}")
    i3d = torch.jit.load(filepath).eval().to(device)
    #i3d = torch.nn.DataParallel(i3d)
    return i3d


def get_feats(videos, detector, device, bs=10):
    # videos : torch.tensor BCTHW [0, 1]
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.
    feats = np.empty((0, 400))
    device = device if device is not torch.device("cpu") else device
    with torch.no_grad():
        for i in range((len(videos)-1)//bs + 1):
            feats = np.vstack([feats, detector(torch.stack([preprocess_single(video) for video in videos[i*bs:(i+1)*bs]]).to(device), **detector_kwargs).detach().cpu().numpy()])
    return feats


def get_fvd_feats(videos, i3d, device, bs=10):
    # videos in [0, 1] as torch tensor BCTHW
    # videos = [preprocess_single(video) for video in videos]
    embeddings = get_feats(videos, i3d, device, bs)
    return embeddings



def preprocess_single(video, resolution=224, sequence_length=None):
    # video: CTHW, [0, 1]
    c, t, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:, :sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear', align_corners=False)

    # center crop
    c, t, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]

    # [0, 1] -> [-1, 1]
    video = (video - 0.5) * 2

    return video.contiguous()


def get_logits(i3d, videos, device):
    #assert videos.shape[0] % 2 == 0
    logits = torch.empty(0, 400)
    with torch.no_grad():
        for i in range(len(videos)):
            # logits.append(i3d(preprocess_single(videos[i]).unsqueeze(0).to(device)).detach().cpu())
            logits = torch.vstack([logits, i3d(preprocess_single(videos[i]).unsqueeze(0).to(device)).detach().cpu()])
    # logits = torch.cat(logits, dim=0)
    return logits


def get_fvd_logits(videos, i3d, device):
    # videos in [0, 1] as torch tensor BCTHW
    # videos = [preprocess_single(video) for video in videos]
    embeddings = get_logits(i3d, videos, device)
    return embeddings


# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L161
def _symmetric_matrix_square_root(mat, eps=1e-10):
    u, s, v = torch.linalg.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())


# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L400
def trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()

    fact = 1.0 / (m.size(1) - 1) # unbiased estimate
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


# def frechet_distance(x1, x2):
#     x1 = x1.flatten(start_dim=1)
#     x2 = x2.flatten(start_dim=1)
#     m, m_w = x1.mean(dim=0), x2.mean(dim=0)
#     sigma, sigma_w = cov(x1, rowvar=False), cov(x2, rowvar=False)
#     sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)
#     trace = torch.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component
#     mean = torch.sum((m - m_w) ** 2)
#     fd = trace + mean
#     return fd


"""
Copy-pasted from https://github.com/cvpr2022-stylegan-v/stylegan-v/blob/main/src/metrics/frechet_video_distance.py
"""
from typing import Tuple
from scipy.linalg import sqrtm
import numpy as np


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0) # [d]
    sigma = np.cov(feats, rowvar=False) # [d, d]
    return mu, sigma


def frechet_distance(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)
    m = np.square(mu_gen - mu_real).sum()
    sigma_gen = np.atleast_2d(sigma_gen)
    sigma_real = np.atleast_2d(sigma_real)
    s, _ = sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)




def calculate_fvd(video1,video2):
    i3d = load_i3d_pretrained(video1.device)
    a_f = get_fvd_feats(video1, i3d, video1.device, bs=10)
    b_f = get_fvd_feats(video2, i3d, video2.device, bs=10)
    fvd = frechet_distance(a_f,b_f)
    return fvd


def calc_metric(video1, video2):
    """Calculate PSNR and SSIM for images.
        img1: ndarray, range [0, 1]
        img2: ndarray, range [0, 1]

    """
    loss_fn_alex = lpips.LPIPS(net='alex').to(video1.device) # best forward scores
    #loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
    
    num_frames = video1.shape[2]
    psnr_list=[]
    ssim_list=[]
    lpips_list=[]
    for i in range(2, num_frames):
        img1 = video1[:,:,i,:,:].clone()
        img2 = video2[:,:,i,:,:].clone()
        lpips_list.append(loss_fn_alex(img1, img2))
        img1=img1.squeeze().cpu().numpy()
        img2=img2.squeeze().cpu().numpy()
        psnr_list.append(compare_psnr(img1,img2, data_range=1))
        ssim_list.append(compare_ssim(img1,img2, data_range=1,channel_axis=0))

    psnr = sum(psnr_list)/len(psnr_list)
    ssim = sum(ssim_list)/len(ssim_list)
    avg_lpips = sum(lpips_list)/len(lpips_list)
    #fvd = calculate_fvd(video1,video2)

    return psnr, ssim, avg_lpips.item()#, fvd

if __name__=='__main__':
    a=torch.randn((1,3,30,256,256),device='cuda:4')
    b=torch.randn((1,3,30,256,256),device='cuda:4')
    i3d = load_i3d_pretrained(a.device)
    
    a_f = get_fvd_feats(a, i3d, a.device, bs=10)
    b_f = get_fvd_feats(b, i3d, b.device, bs=10)
    fvd = frechet_distance(a_f,b_f)