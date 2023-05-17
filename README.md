<div align="center">    

## Video Diffusion Models with Local-Global Context Guidance

[![Arxiv](http://img.shields.io/badge/Arxiv-2302.10663-B31B1B.svg)](https://arxiv.org/abs/2302.10663)
[![Conference](http://img.shields.io/badge/CVPR-2023-4b44ce.svg)](https://arxiv.org/abs/2302.10663)
</div>


## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
  * [Abstract](#abstract)
  * [Method](#method)
  * [Examples](#examples)
- [Running the code](#running-the-code)
  * [Dependencies](#dependencies)
  * [Data](#data)
  * [Textual Inversion](#textual-inversion)
  * [Side note: Textual Inversion Initialization](#side-note--textual-inversion-initialization)
  * [Reconstruction](#reconstruction)
  * [Examples](#examples-1)
  * [Extra tips](#extra-tips)
  * [Pretrained checkpoints](#pretrained-checkpoints)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)


## Overview

### Abstract
Diffusion models have emerged as a powerful paradigm in video synthesis tasks including prediction, generation, and interpolation. Due to the limitation of the computational budget, existing methods usually implement conditional diffusion models with an autoregressive inference pipeline, in which the future fragment is predicted based on the distribution of adjacent past frames. However, only the conditions from a few previous frames can't capture the global temporal coherence, leading to inconsistent or even outrageous results in long-term video prediction.  
In this paper, we propose a Local-Global Context guided Video Diffusion model (LGC-VD) to capture multi-perception conditions for producing high-quality videos in both conditional/unconditional settings. In LGC-VD, the UNet is implemented with stacked residual blocks with self-attention units, avoiding the undesirable computational cost in 3D Conv. We construct a local-global context guidance strategy to capture the multi-perceptual embedding of the past fragment to boost the consistency of future prediction. Furthermore, we propose a two-stage training strategy to alleviate the effect of noisy frames for more stable predictions. Our experiments demonstrate that the proposed method achieves favorable performance on video prediction, interpolation, and unconditional video generation. 
### Method
<div align=center><img src="assets/diagram.png"></div>

### EXamples
#### Prediciton results on Cityscapes
<div align=center><img src="assets/city_prediction.gif"></div>

From top to bottom: Ground Truth (Top Row), MCVD spatin (Second Row), MCVD concat (Third Row), Our Method (Bottom Row).
#### Prediciton results on BAIR
<div align=center><img src="assets/bair_prediction.gif"></div>

From top to bottom: Ground Truth (Top Row), MCVD spatin (Second Row), MCVD concat (Third Row), Our Method (Bottom Row).
#### unconditional generation results on BAIR
<div align=center><img src="assets/bair_generation.gif"></div>

Our unconditional generation results on BAIR.
#### video infilling results on BAIR
<div align=center><img src="assets/bair_infilling.png"></div>

Our video infilling results on BAIR.

## Running the code
### Data preparation
#### Cityscapes
1. Download Cityscapes video dataset (leftImg8bit_sequence_trainvaltest.zip (324GB)) from  https://www.cityscapes-dataset.com/
2. Convert it to HDF5 format, and save in /path/to/Cityscape_h5:
'''
python datasets/cityscapes_convert.py --leftImg8bit_sequence_dir '/path/to/Cityscapes/leftImg8bit_sequence' --image_size 128 --out_dir '/path/to/Cityscapes_h5' 
'''
### Training

### Testing


## Acknowledgement
Our work is based on https://github.com/lucidrains/denoising-diffusion-pytorch


