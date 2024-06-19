# [CVPR 2024] Generative Unlearning for Any Identity

Official PyTorch implementation for CVPR 2024 paper:

**Generative Unlearning for Any Identity**  
[Juwon Seo](https://github.com/JJuOn)\*, [Sung-Hoon Lee](https://github.com/Ethan-Lee-Sunghoon)\*, [Tae-Young Lee](https://github.com/TY-LEE-KR)\*, [Seungjun Moon](https://seungjun-moon.github.io/tabs/about.html), and Gyeong-Moon Park<sup>$\dagger$</sup>   

[![arXiv](https://img.shields.io/badge/arXiv-2405.09879-b31b1b.svg)](https://arxiv.org/abs/2405.09879) 

# Environment
- Python 3.9.x
- PyTorch 2.1.0
- Torchvision 0.16.0
- NVIDIA GeForce RTX 3090 / A5000 / A6000
- CUDA 11.8


# Getting Started
## Environment
```bash
git clone git@github.com/KHU-AGI/GUIDE.git
cd GUIDE
conda env create -f environment.yaml
conda activate guide
```
## Files to Run Code
|Filename|Description|
|-|-|
|ffhqrebalanced512-128.pkl|Pretrained weights of EG3D network|
|w_avg_ffhqrebalanced512-128.pt|Average latent code computed from pretrained EG3D network|
|model_ir_se50.pth|Pretrained weights of ArcFace network to compute identity loss|
|encoder_FFHQ.pt|Pretrained weights of GOAEncoder for 3D GAN inversion|
|CurricularFace_Backbone.pth|Pretrained weights of CurricularFace network to compute identity similarity|

To run code, please download all of files via [Google Drive](https://drive.google.com/drive/folders/1tl7zLPZgwOpa6xWmRsjz7LMrysfSbh_8?usp=drive_link).

# Erase an identity from pretrained 3D GAN
We provide sample data (./data/CelebAHQ/512) to run our code.  
The commands for Random and In-Domain scenarios will be updated in near future.
```bash
# Baseline
python unlearn.py --exp baseline \
                  --inversion goae \
                  --inversion_image_path ./data/CelebAHQ/512 \
                  --target average \
                  --local \
                  --seed 0

# GUIDE
python unlearn.py --exp guide \
                  --inversion goae \
                  --inversion_image_path ./data/CelebAHQ/512 \
                  --target extra \
                  --target_d 30.0 \
                  --local \
                  --adj \
                  --glob \
                  --seed 0
```
It takes about 15 minutes in a single 3090 GPU.
## Erase in the wild identities
If you want to erasure in the wild identities, please preprocess the images via [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch).  
Otherwise, both the inversion encoder and pretrained generatior can't not recognize them correctly. 
# Evaluation
Evaluation on identity erasure:
```bash
python evaluate_id.py --exp [baseline | guide]
```
Evaluation on pretrained distribution preservation:
```bash
python evaluate_fid.py --exp [baseline | guide]
```
By running the above commands, we could obtain the results of:  
| |ID|ID<sub>others</sub>|FID<sub>pre</sub>|ΔFID<sub>real</sub>|
|-|-|-|-|-|
|baseline|0.0818|0.4122|9.487|4.891|
|GUIDE|0.0183|0.2797|7.497|3.331|

# Acknowledgement
Our code is based on [EG3D](https://github.com/NVlabs/eg3d), [GOAE](https://github.com/jiangyzy/GOAE), [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch), [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity), [pytorch-fid](https://github.com/mseitzer/pytorch-fid), [InsightFace](https://github.com/TreB1eN/InsightFace_Pytorch), and [CurricularFace](https://github.com/HuangYG123/CurricularFace).
# BibTex
```
@inproceedings{seo2024generative,
    author    = {Seo, Juwon and Lee, Sung-Hoon and Lee, Tae-Young and Moon, Seungjun and Park, Gyeong-Moon},
    title     = {Generative Unlearning for Any Identity},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {9151-9161}
}
```