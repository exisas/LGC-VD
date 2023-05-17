import numpy as np
import torch 

from einops import rearrange
from video_diffusion_pytorch import GaussianDiffusion
import torch.nn.functional as F
def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f"cannot create exactly {num_timesteps} steps with an integer stride")
        elif section_counts == "fast27":
            steps = space_timesteps(num_timesteps, "10,10,3,2,2")
            # Help reduce DDIM artifacts from noisiest timesteps.
            steps.remove(num_timesteps - 1)
            steps.add(num_timesteps - 3)
            return steps
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, betas, encoder_fn, denoise_fn, image_size, num_frames, rank, timesteps, loss_type ):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(betas)
        betas = torch.tensor(betas, dtype=torch.float32).to(rank)
        base_diffusion = GaussianDiffusion(betas=betas,
                                encoder_fn=encoder_fn,
                                denoise_fn=denoise_fn, 
                                image_size=image_size,
                                num_frames=num_frames,
                                channels = 3,
                                timesteps=timesteps,
                                loss_type=loss_type)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        betas = torch.tensor(new_betas, dtype=torch.float32).to(rank)
        super().__init__(betas=betas,
                                encoder_fn=encoder_fn,
                                denoise_fn=denoise_fn, 
                                image_size=image_size,
                                num_frames=num_frames,
                                channels = 3,
                                timesteps=timesteps,
                                loss_type=loss_type)


    # def p_mean_variance(self, *args, **kwargs):
    #     return super().p_mean_variance(self._wrap_model(self.denoise_fn), *args, **kwargs)

    
    def p_mean_variance(self, x, t, clip_denoised: bool, cond = None, cond_feature=None, cond_scale = 1.):
        map_tensor = torch.tensor(self.timestep_map, device=t.device, dtype=t.dtype)
        new_ts = map_tensor[t]

        #x_recon = self.predict_start_from_noise(x, t=t, noise = self.denoise_fn.forward_with_cond_scale(x, new_ts, cond = cond, cond_scale = cond_scale))
        x_recon = self.v_predict_start_from_noise(x, t=t, v_predict=self.denoise_fn.forward_with_cond_scale(x, new_ts, cond = cond,cond_feature=cond_feature, cond_scale = cond_scale))

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim = -1
                )

                s.clamp_(min = 1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance


