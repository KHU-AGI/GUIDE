import dnnlib
import click
import legacy
import os
import torch
import random
import scipy
import torchvision

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datetime import datetime
try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=(DEFAULT_BLOCK_INDEX,),
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False,
                 use_fid_inception=True):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = _inception_v3(weights='DEFAULT')

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def _inception_v3(*args, **kwargs):
    """Wraps `torchvision.models.inception_v3`"""
    try:
        version = tuple(map(int, torchvision.__version__.split('.')[:2]))
    except ValueError:
        # Just a caution against weird version strings
        version = (0,)

    # Skips default weight inititialization if supported by torchvision
    # version. See https://github.com/mseitzer/pytorch-fid/issues/28.
    if version >= (0, 6):
        kwargs['init_weights'] = False

    # Backwards compatibility: `weights` argument was handled by `pretrained`
    # argument prior to version 0.13.
    if version < (0, 13) and 'weights' in kwargs:
        if kwargs['weights'] == 'DEFAULT':
            kwargs['pretrained'] = True
        elif kwargs['weights'] is None:
            kwargs['pretrained'] = False
        else:
            raise ValueError(
                'weights=={} not supported in torchvision {}'.format(
                    kwargs['weights'], torchvision.__version__
                )
            )
        del kwargs['weights']

    return torchvision.models.inception_v3(*args, **kwargs)


def fid_inception_v3():
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = _inception_v3(num_classes=1008,
                              aux_logits=False,
                              weights=None)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)

    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class FIDInceptionA(torchvision.models.inception.InceptionA):
    """InceptionA block patched for FID computation"""
    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(torchvision.models.inception.InceptionC):
    """InceptionC block patched for FID computation"""
    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""
    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: The FID Inception model uses max pooling instead of average
        # pooling. This is likely an error in this specific Inception
        # implementation, as other Inception models use average pooling here
        # (which matches the description in the paper).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

torch.set_grad_enabled(False)

class ImageDataset(Dataset):
    def __init__(self, path, transform=None, num_samples=None):
        self.path = path
        self.transform = transform

        self.imgs = sorted(os.listdir(self.path))
        if num_samples is not None:
            self.imgs = self.imgs[:num_samples]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.path, self.imgs[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

def extract_features(inception, loader, device="cuda"):
    feats = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            feat = inception(data)[0].squeeze(3).squeeze(2).cpu().numpy()
            feats.append(feat)
    
    return np.concatenate(feats, axis=0)

def calculate_statistics(feats):
    m = np.mean(feats, axis=0)
    s = np.cov(feats, rowvar=False)
    return m, s

def calculate_fid(feats1, feats2):
    m1, s1 = calculate_statistics(feats1)
    m2, s2 = calculate_statistics(feats2)

    m1, m2 = np.atleast_1d(m1), np.atleast_1d(m2)
    s1, s2 = np.atleast_2d(s1), np.atleast_2d(s2)

    diff = m1 - m2
    covmean, _ = scipy.linalg.sqrtm(s1.dot(s2), disp=False)

    if not np.isfinite(covmean).all():
        eps = 1e-6
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(s1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((s1 + offset).dot(s2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    return diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean)

def convert_tensor(t):
    t = (t.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(t[0].cpu().numpy(), "RGB")

@click.command()
@click.option("--pretrained_ckpt", type=str, default="ffhqrebalanced512-128.pkl")
@click.option("--ckpt", type=str, default=None)
@click.option("--exp", type=str, required=True)
@click.option("--num_samples", type=int, default=5000)
@click.option("--batch_size", type=int, default=50)
@click.option("--seed", type=int, default=0)
@click.option("--angle_p", type=float, default=-0.2)
@click.option("--angle_y_abs", type=float, default=np.pi / 12)
@click.option("--truncation", type=float, default=1)
@click.option("--truncation_cutoff", type=int, default=14)
def main(pretrained_ckpt, ckpt, exp, num_samples, batch_size, seed, angle_p, angle_y_abs, truncation, truncation_cutoff):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    fake_image_ours_path = f"fake_images/{exp}"
    fake_image_pretrained_path = f"fake_images/pretrained/{seed}"

    if os.path.exists(fake_image_pretrained_path) and len(os.listdir(fake_image_pretrained_path)) == num_samples:
        print(f"Skipping generation from pretrained model as it already exists")
        skip_pretrained = True

    os.makedirs(fake_image_ours_path, exist_ok=True)
    if not skip_pretrained:
        os.makedirs(fake_image_pretrained_path, exist_ok=True)

    with dnnlib.util.open_url(pretrained_ckpt) as f:
        source_G = legacy.load_network_pkl(f)["G_ema"].cuda()

    if ckpt is None:
        ckpt = os.path.join("experiments", exp, "checkpoints", "last.pkl")
    with dnnlib.util.open_url(ckpt) as f:
        G = legacy.load_network_pkl(f)["G_ema"].cuda()

    source_G.eval()
    G.eval()

    angle_p = -0.2
    intrinsics = FOV_to_intrinsics(18.837).cuda()
    cam_pivot = torch.tensor(G.rendering_kwargs.get("avg_cam_pivot", [0, 0, 0]), device="cuda")
    cam_radius = G.rendering_kwargs.get("avg_cam_radius", 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device="cuda")

    start_time = datetime.now()

    for i in range(num_samples):
        z = torch.randn(1, G.z_dim).cuda()
        angle_y = np.random.uniform(-angle_y_abs, angle_y_abs)
        cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device="cuda")
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1 ,16), intrinsics.reshape(-1, 9)], dim=1)

        if not skip_pretrained:
            w = source_G.mapping(z, conditioning_params, truncation_psi=truncation, truncation_cutoff=truncation_cutoff)
            img = source_G.synthesis(w, camera_params)["image"]
            img = convert_tensor(img)
            img.save(os.path.join(fake_image_pretrained_path, f"{i}.png"))
            del img

        w = G.mapping(z, conditioning_params, truncation_psi=truncation, truncation_cutoff=truncation)
        img_ours = G.synthesis(w, camera_params)["image"]
        img_ours = convert_tensor(img_ours)
        img_ours.save(os.path.join(fake_image_ours_path, f"{i}.png"))

        del z, w, img_ours

        if i % 100 == 0:
            print(f"Generated {i + 1} images in {datetime.now() - start_time}")

    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).cuda()
    inception.eval()

    transform = transforms.ToTensor()

    source_data = ImageDataset(fake_image_pretrained_path, transform=transform, num_samples=num_samples)
    source_loader = DataLoader(source_data, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)

    our_data = ImageDataset(fake_image_ours_path, transform=transform, num_samples=num_samples)
    our_loader = DataLoader(our_data, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)
    
    feats1 = extract_features(inception, source_loader)
    feats2 = extract_features(inception, our_loader)

    fid_pre = calculate_fid(feats1, feats2)

    ffhq_feat_path = "ffhq_real_feat.npy"
    if os.path.exists(ffhq_feat_path):
        feats3 = np.load(ffhq_feat_path)
        
    fid2 = calculate_fid(feats3, feats1)
    fid3 = calculate_fid(feats3, feats2)
    fid_real = fid3 - fid2

    print(f"FID_pre: {fid_pre} \n FID_real: {fid_real}")

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter