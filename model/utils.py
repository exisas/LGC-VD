import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import string


def _einsum(a, b, c, x, y):
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  return torch.einsum(einsum_str, x, y)

def contract_inner(x, y):
  """tensordot(x, y, 1)."""
  x_chars = list(string.ascii_lowercase[:len(x.shape)])
  y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x.shape)])
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)



class NIN(nn.Module):
  def __init__(self, in_dim, num_units, init_scale=0.1):
    super().__init__()
    self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

  def forward(self, x):
    x = x.permute(0, 2, 3, 1)
    y = contract_inner(x, self.W) + self.b
    return y.permute(0, 3, 1, 2)





def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """Ported from JAX. """

  def _compute_fans(shape, in_axis=1, out_axis=0):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init

def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')


def ddpm_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
  """3x3 convolution with DDPM initialization."""
  conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                   dilation=dilation, bias=bias)
  conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
  nn.init.zeros_(conv.bias)
  return conv

conv3x3 = ddpm_conv3x3

class ResnetBlockDDPMpp(nn.Module):
  """ResBlock adapted from DDPM."""

  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False,
               dropout=0.1, skip_rescale=False, init_scale=0., n_frames=1, 
               act3d=False):
    super().__init__()

    conv3x3_ = conv3x3
  

    out_ch = out_ch if out_ch else in_ch
    num_groups = min(in_ch // 4, 32)
    while (in_ch % num_groups != 0):
      num_groups -= 1
    self.GroupNorm_0 = nn.GroupNorm(num_groups=num_groups, num_channels=in_ch, eps=1e-6)
    self.Conv_0 = conv3x3_(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)
    num_groups = min(out_ch // 4, 32)
    while (in_ch % num_groups != 0):
      num_groups -= 1
    self.GroupNorm_1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3_(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch:
      if conv_shortcut:
        self.Conv_2 = conv3x3_(in_ch, out_ch)
      else:
        self.NIN_0 = NIN(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.out_ch = out_ch
    self.conv_shortcut = conv_shortcut

  def forward(self, x, temb=None):
    h = self.act(self.GroupNorm_0(x).float())
    h = self.Conv_0(h)
    if temb is not None:
      h += self.Dense_0(self.act(temb.float()))[:, :, None, None]
    h = self.act(self.GroupNorm_1(h).float())
    h = self.Dropout_0(h)
    h = self.Conv_1(h)
    if x.shape[1] != self.out_ch:
      if self.conv_shortcut:
        x = self.Conv_2(x)
      else:
        x = self.NIN_0(x)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)
    
    


# Added multi-head attention similar to https://github.com/openai/guided-diffusion/blob/912d5776a64a33e3baf3cff7eb1bcba9d9b9354c/guided_diffusion/unet.py#L361
class AttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""

  def __init__(self, channels, skip_rescale=False, init_scale=0., n_heads=1, n_head_channels=-1):
    super().__init__()
    num_groups = min(channels // 4, 32)
    while (channels % num_groups != 0): # must find another value
        num_groups -= 1
    self.GroupNorm_0 = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6)
    #self.Norm_0 = Norm(dim=channels)
    #self.act = nn.SiLU()
    self.NIN_0 = NIN(channels, channels)
    self.NIN_1 = NIN(channels, channels)
    self.NIN_2 = NIN(channels, channels)
    self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
    self.skip_rescale = skip_rescale
    if n_head_channels == -1:
        self.n_heads = n_heads
    else:
        if channels < n_head_channels:
          self.n_heads = 1
        else:
          assert channels % n_head_channels == 0
          self.n_heads = channels // n_head_channels
    
  def forward(self, x):
    B, C, H, W = x.shape
    h = self.GroupNorm_0(x)
    #h = self.act(x)
    q = self.NIN_0(h)
    k = self.NIN_1(h)
    v = self.NIN_2(h)

    C = C // self.n_heads

    w = torch.einsum('bchw,bcij->bhwij', q.reshape(B * self.n_heads, C, H, W), k.reshape(B * self.n_heads, C, H, W)) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B * self.n_heads, H, W, H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B * self.n_heads, H, W, H, W))
    h = torch.einsum('bhwij,bcij->bchw', w, v.reshape(B * self.n_heads, C, H, W))
    h = h.reshape(B, C * self.n_heads, H, W)
    h = self.NIN_3(h)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)



class AttnBlock_cross(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""

  def __init__(self, channels, skip_rescale=False, init_scale=0., n_heads=1, n_head_channels=-1):
    super().__init__()
    num_groups = min(channels // 4, 32)
    while (channels % num_groups != 0): # must find another value
        num_groups -= 1
    self.GroupNorm_0 = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6)
    #self.Norm_0 = Norm(dim=channels)
    #self.act = nn.SiLU()
    self.NIN_0 = NIN(channels, channels)
    self.NIN_1 = NIN(channels, channels)
    self.NIN_2 = NIN(channels, channels)
    self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
    self.skip_rescale = skip_rescale
    if n_head_channels == -1:
        self.n_heads = n_heads
    else:
        if channels < n_head_channels:
          self.n_heads = 1
        else:
          assert channels % n_head_channels == 0
          self.n_heads = channels // n_head_channels
    
  def forward(self, x, cond_feature):
    B, C, H, W = x.shape

    h = self.GroupNorm_0(x)
    f = self.GroupNorm_0(cond_feature)
    #h = self.act(x)
    q = self.NIN_0(h)
    k = self.NIN_1(f)
    v = self.NIN_2(f)

    C = C // self.n_heads

    w = torch.einsum('bchw,bcij->bhwij', q.reshape(B * self.n_heads, C, H, W), k.reshape(B * self.n_heads, C, H, W)) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B * self.n_heads, H, W, H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B * self.n_heads, H, W, H, W))
    h = torch.einsum('bhwij,bcij->bchw', w, v.reshape(B * self.n_heads, C, H, W))
    h = h.reshape(B, C * self.n_heads, H, W)
    h = self.NIN_3(h)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)
    
    
    

def get_norm(norm, ch, affine=True):
  """Get activation functions from the opt file."""
  if norm == 'none':
    return nn.Identity()
  elif norm == 'batch':
    return nn.BatchNorm1d(ch, affine = affine)
  elif norm == 'evo':
    return EvoNorm2D(ch = ch, affine = affine, eps = 1e-5, groups = min(ch // 4, 32))
  elif norm == 'group':
    num_groups=min(ch // 4, 32)
    while(ch % num_groups != 0): # must find another value
      num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=ch, eps=1e-5, affine=affine)
  elif norm == 'layer':
    return nn.LayerNorm(normalized_shape=ch, eps=1e-5, elementwise_affine=affine)
  elif norm == 'instance':
    return nn.InstanceNorm2d(num_features=ch, eps=1e-5, affine=affine)
  else:
    raise NotImplementedError('norm choice does not exist')


class get_act_norm(nn.Module): # order is norm -> act
  def __init__(self, act, act_emb, norm, ch, emb_dim=None, spectral=False, is3d=False, n_frames=1, cond_ch=0):
    super(get_act_norm, self).__init__()
    
    self.norm = norm
    self.act = act
    self.act_emb = act_emb
    self.is3d = is3d
    self.n_frames = n_frames
    self.cond_ch = cond_ch
    if emb_dim is not None:
      if self.is3d:
        out_dim = 2*(ch // self.n_frames)
      else:
        out_dim = 2*ch
      if spectral:
        self.Dense_0 = torch.nn.utils.spectral_norm(nn.Linear(emb_dim, out_dim))
      else:
        self.Dense_0 = nn.Linear(emb_dim, out_dim)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
      nn.init.zeros_(self.Dense_0.bias)
      affine = False # We remove scale/intercept after normalization since we will learn it with [temb, yemb]
    else:
      affine = True

    self.Norm_0 = get_norm(norm, (ch // n_frames) if is3d else ch, affine)

  def forward(self, x, emb=None, cond=None):
    if emb is not None:
      #emb = torch.cat([temb, yemb], dim=1) # Combine embeddings
      emb_out = self.Dense_0(self.act_emb(emb.float()))[:, :, None, None] # Linear projection
      # ada-norm as in https://github.com/openai/guided-diffusion
      scale, shift = torch.chunk(emb_out, 2, dim=1)
      if self.is3d:
        B, CN, H, W = x.shape
        N = self.n_frames
        scale = scale.reshape(B, -1, 1, 1, 1)
        shift = shift.reshape(B, -1, 1, 1, 1)
        x = x.reshape(B, -1, N, H, W)
      if self.norm == 'spade':
        emb_norm = self.Norm_0(x, cond)
        emb_norm = emb_norm.reshape(B, -1, N, H, W) if self.is3d else emb_norm
      else:
        emb_norm = self.Norm_0(x)
      x = emb_norm * (1 + scale) + shift
      if self.is3d:
        x = x.reshape(B, -1, H, W)
    else:
      if self.is3d:
        B, CN, H, W = x.shape
        N = self.n_frames
        x = x.reshape(B, -1, N, H, W)
      if self.norm == 'spade':
        x = self.Norm_0(x, cond)
      else:
        x = self.Norm_0(x)
        x = x.reshape(B, CN, H, W) if self.is3d else x
    x = self.act(x.float())
    return(x)
