import os
import click
import legacy
import dnnlib
import torch
import random
import pickle
import copy
import lpips

import torch.nn.functional as F
import numpy as np

from torch import optim
from tqdm import tqdm
from training.triplane import TriPlaneGenerator
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils.misc import copy_params_and_buffers
from PIL import Image
from arcface import IDLoss

def tensor_to_image(t):
    t = (t.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(t[0].cpu().numpy(), "RGB")

def image_to_tensor(i, size=256):
    i = i.resize((size, size))
    i = np.array(i)
    i = i.transpose(2, 0, 1)
    i = torch.from_numpy(i).to(torch.float32).to("cuda") / 127.5 - 1
    return i

@click.command()
@click.option("--pretrained_ckpt", type=str, default="ffhqrebalanced512-128.pkl")
@click.option("--iter", type=int, default=1000)
@click.option("--lr", type=float, default=1e-4)
@click.option("--seed", type=int, default=None)
@click.option("--fov-deg", type=float, default=18.837)
@click.option("--truncation_psi", type=float, default=1.0)
@click.option("--truncation_cutoff", type=int, default=14)
@click.option("--exp", type=str, required=True)
@click.option("--inversion", type=str, default=None)
@click.option("--inversion_image_path", type=str, default=None)
@click.option("--angle_p", type=float, default=-0.2)
@click.option("--angle_y_abs", type=float, default=np.pi / 12)
@click.option("--sample_views", type=int, default=11)
# latent target unlearning: local unlearning loss
@click.option("--local", is_flag=True)
@click.option("--loss_local_mse_lambda", type=float, default=1e-2)
@click.option("--loss_local_lpips_lambda", type=float, default=1.0)
@click.option("--loss_local_id_lambda", type=float, default=0.1)
# latent target unlearning: adjacency-aware unlearning loss
@click.option("--adj", is_flag=True)
@click.option("--loss_adj_mse_lambda", type=float, default=1e-2)
@click.option("--loss_adj_lpips_lambda", type=float, default=1.0)
@click.option("--loss_adj_id_lambda", type=float, default=0.1)
@click.option("--loss_adj_batch", type=int, default=2)
@click.option("--loss_adj_lambda", type=float, default=1.0)
@click.option("--loss_adj_alpha_range_min", type=int, default=0)
@click.option("--loss_adj_alpha_range_max", type=int, default=15)
# latent target unlearning: global preservation loss
@click.option("--glob", is_flag=True)
@click.option("--loss_global_lambda", type=float, default=1.0)
@click.option("--loss_global_batch", type=int, default=2)
# latent target unlearning: un-identifying face on latent space
@click.option("--target_idx", type=int, default=0)
@click.option("--target", type=str, default="extra")
@click.option("--target_d", type=float, default=30.0)
def unlearn(*args, **kwargs):
    pretrained_ckpt = kwargs["pretrained_ckpt"]
    iter = kwargs["iter"]
    lr = kwargs["lr"]
    seed = kwargs["seed"]
    fov_deg = kwargs["fov_deg"]
    truncation_psi = kwargs["truncation_psi"]
    truncation_cutoff = kwargs["truncation_cutoff"]
    exp = kwargs["exp"]
    inversion = kwargs["inversion"]
    inversion_image_path = kwargs["inversion_image_path"]

    angle_p = kwargs["angle_p"]
    angle_y_abs = kwargs["angle_y_abs"]
    sample_views = kwargs["sample_views"]

    local = kwargs["local"]
    loss_local_mse_lambda = kwargs["loss_local_mse_lambda"]
    loss_local_lpips_lambda = kwargs["loss_local_lpips_lambda"]
    loss_local_id_lambda = kwargs["loss_local_id_lambda"]

    adj = kwargs["adj"]
    loss_adj_mse_lambda = kwargs["loss_adj_mse_lambda"]
    loss_adj_lpips_lambda = kwargs["loss_adj_lpips_lambda"]
    loss_adj_id_lambda = kwargs["loss_adj_id_lambda"]
    loss_adj_batch = kwargs["loss_adj_batch"]
    loss_adj_lambda = kwargs["loss_adj_lambda"]
    loss_adj_alpha_range_min = kwargs["loss_adj_alpha_range_min"]
    loss_adj_alpha_range_max = kwargs["loss_adj_alpha_range_max"]

    glob = kwargs["glob"]
    loss_global_lambda = kwargs["loss_global_lambda"]
    loss_global_batch = kwargs["loss_global_batch"]

    target_idx = kwargs["target_idx"]
    target = kwargs["target"]
    target_d = kwargs["target_d"]


    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
    print(exp)

    device = torch.device("cuda")
    with dnnlib.util.open_url(pretrained_ckpt) as f:
        g_source = legacy.load_network_pkl(f)["G_ema"].to(device)
    
    generator = TriPlaneGenerator(*g_source.init_args, **g_source.init_kwargs).requires_grad_(False).to(device)
    copy_params_and_buffers(g_source, generator, require_all=True)
    generator.neural_rendering_resolution = g_source.neural_rendering_resolution
    generator.rendering_kwargs = g_source.rendering_kwargs
    generator.load_state_dict(g_source.state_dict(), strict=False)
    generator.train()

    g_source = copy.deepcopy(generator)

    for name, param in g_source.named_parameters():
        param.requires_grad = False

    for name, param in generator.named_parameters():
        if "backbone.synthesis" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    exp_dir = f"experiments/{exp}"
    ckpt_dir = f"experiments/{exp}/checkpoints"
    image_dir = f"experiments/{exp}/training/images"
    result_dir = f"experiments/{exp}/training/results"

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "args.txt"), "w") as f:
        for arg in kwargs:
            f.write(f"{arg}: {kwargs[arg]}\n")

    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam_pivot = torch.tensor(generator.rendering_kwargs.get("avg_cam_pivot", [0, 0, 0]), device=device)
    cam_radius = generator.rendering_kwargs.get("avg_cam_radius", 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device=device)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
    front_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2 - 0.2, cam_pivot, radius=cam_radius, device=device)
    camera_params_front = torch.cat([front_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
    
    optimizer = optim.Adam(generator.parameters(), lr=lr)

    w_avg = torch.load("w_avg_ffhqrebalanced512-128.pt", map_location=device).unsqueeze(0) # [1, 14, 512]

    # Visualize before unlearning
    with torch.no_grad():
        if inversion is not None:
            assert inversion_image_path is not None, "The path of an image to invert is required."
            assert inversion in ["goae"]
            if inversion == "goae":
                from goae import GOAEncoder
                from goae.swin_config import get_config
                
                swin_config = get_config()
                stage_list = [10000, 20000, 30000]
                encoder_ckpt = "encoder_FFHQ.pt"

                encoder = GOAEncoder(swin_config=swin_config, mlp_layer=2, stage_list=stage_list).to(device)
                encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
                if os.path.isdir(inversion_image_path):
                    filenames = sorted(os.listdir(inversion_image_path))
                    imgs = [image_to_tensor(Image.open(os.path.join(inversion_image_path, filename)).convert("RGB")) for filename in filenames]
                    imgs = torch.stack(imgs, dim=0)
                    w, _ = encoder(imgs)
                    w_origin = w + w_avg
                    w_u = w[[target_idx], :, :] + w_avg
                    del imgs
                else:
                    img = image_to_tensor(Image.open(inversion_image_path).convert("RGB")).unsqueeze(0)
                    w, _ = encoder(img)
                    w_u = w + w_avg
                    del img
            else:
                raise NotImplementedError
        else:
            w_avg = torch.load("w_avg_ffhqrebalanced512-128.pt", map_location=device).unsqueeze(0) # [1, 14, 512]
            if inversion_image_path is not None:
                w_u = torch.load(inversion_image_path)
            else:
                z_u = torch.randn(1, 512, device=device)
                w_u = generator.mapping(z_u, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

        generator.eval()
        if inversion is None: # for random
            for view, angle_y in enumerate(np.linspace(-angle_y_abs, angle_y_abs, sample_views)):
                cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                camera_params_view = torch.cat([cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
                img_u = generator.synthesis(w_u, camera_params_view)["image"]
                img_u = tensor_to_image(img_u)
                img_u.save(os.path.join(result_dir, f"unlearn_before_0_{view}.png"))
            del img_u
        else:
            if os.path.isdir(inversion_image_path): # for OOD
                for i in range(len(filenames)):
                    for view, angle_y in enumerate(np.linspace(-angle_y_abs, angle_y_abs, sample_views)):
                        cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                        camera_params_view = torch.cat([cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
                        img_origin = generator.synthesis(w_origin[[i]], camera_params_view)["image"]
                        img_origin = tensor_to_image(img_origin)
                        img_origin.save(os.path.join(result_dir, f"unlearn_before_{i}_{view}.png"))
                del img_origin
            else: # for InD
                for view, angle_y in enumerate(np.linspace(-angle_y_abs, angle_y_abs, sample_views)):
                    cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                    camera_params_view = torch.cat([cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
                    img_u = generator.synthesis(w_u, camera_params_view)["image"]
                    img_u = tensor_to_image(img_u)
                    img_u.save(os.path.join(result_dir, f"unlearn_before_0_{view}.png"))
                del img_u
        generator.train()
    
    if target == "average":
        w_target = w_avg
    elif target == "extra":
        with torch.no_grad():
            if inversion is not None:
                w_id = w[[target_idx], :, :]
            else:
                w_id = w_u - w_avg
            w_target = w_avg - w_id / w_id.norm(p=2) * target_d
    

    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    id_fn = IDLoss().to(device)

    pbar = tqdm(range(iter))
    for i in pbar:
        angle_y = np.random.uniform(-angle_y_abs, angle_y_abs)
        cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
        
        loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        # local unlearning loss
        if local:
            loss_local = torch.tensor(0.0, device=device)
            feat_u = generator.get_planes(w_u)
            feat_target = g_source.get_planes(w_target)
            loss_local_mse = F.mse_loss(feat_u, feat_target)
            loss_local = loss_local + loss_local_mse_lambda * loss_local_mse

            img_u = generator.synthesis(w_u, camera_params)["image"]
            img_target = g_source.synthesis(w_target, camera_params)["image"]
            loss_local_lpips = lpips_fn(img_u, img_target).mean()
            loss_local = loss_local + loss_local_lpips_lambda * loss_local_lpips

            loss_local_id = id_fn(img_u, img_target)
            loss_local = loss_local + loss_local_id_lambda * loss_local_id
            loss = loss + loss_local
            loss_dict["loss_local"] = loss_local.item()

        # adjacency-aware unlearning loss
        if adj:
            loss_adj = torch.tensor(0.0, device=device)
            for _ in range(loss_adj_batch):
                z_ra = torch.randn(1, 512, device=device)
                w_ra = generator.mapping(z_ra, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                if loss_adj_alpha_range_max is not None:
                    loss_adj_alpha = torch.from_numpy(np.random.uniform(loss_adj_alpha_range_min, loss_adj_alpha_range_max, size=1)).unsqueeze(1).unsqueeze(1).to(device)
                deltas = loss_adj_alpha * (w_ra - w_u) / (w_ra - w_u).norm(p=2)
                w_u_adj = w_u + deltas
                w_target_adj = w_target + deltas

                feat_u = generator.get_planes(w_u_adj)
                feat_target = g_source.get_planes(w_target_adj)
                loss_adj_mse = F.mse_loss(feat_u, feat_target)
                loss_adj = loss_adj + loss_adj_mse_lambda * loss_adj_mse
            
                img_u = generator.synthesis(w_u_adj, camera_params)["image"]
                img_target = g_source.synthesis(w_target_adj, camera_params)["image"]
                loss_adj_lpips = lpips_fn(img_u, img_target).mean()
                loss_adj = loss_adj + loss_adj_lpips_lambda * loss_adj_lpips

                loss_adj_id = id_fn(img_u, img_target)
                loss_adj = loss_adj + loss_adj_id_lambda * loss_adj_id

            loss = loss + loss_adj_lambda * loss_adj
            loss_dict["loss_adj"] = loss_adj.item()

        # global preservation loss
        if glob:
            loss_global = torch.tensor(0.0, device=device)
            for _ in range(loss_global_batch):
                z_rg = torch.randn(1, 512, device=device)
                w_rg = generator.mapping(z_rg, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                
                img_u = generator.synthesis(w_rg, camera_params)["image"]
                img_target = g_source.synthesis(w_rg, camera_params)["image"]
                loss_global_lpips = lpips_fn(img_u, img_target).mean()
                loss_global = loss_global + loss_global_lpips
            loss = loss + loss_global_lambda * loss_global
            loss_dict["loss_global"] = loss_global.item()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        pbar.set_postfix(loss=loss.item(), **loss_dict)

        if i % 100 == 0:
            with torch.no_grad():
                generator.eval()
                img_u_save = generator.synthesis(w_u, camera_params_front)["image"]
                img_u_save = tensor_to_image(img_u_save)
                img_u_save.save(os.path.join(image_dir, f"img_u_{str(i).zfill(5)}.png"))

                img_target_save = g_source.synthesis(w_target, camera_params_front)["image"]
                img_target_save = tensor_to_image(img_target_save)
                img_target_save.save(os.path.join(image_dir, f"img_target_{str(i).zfill(5)}.png"))
                generator.train()
            del img_u_save, img_target_save

    with torch.no_grad():
        generator.eval()
        img_u_save = generator.synthesis(w_u, camera_params)["image"]
        img_target_save = g_source.synthesis(w_target, camera_params)["image"]
        img_u_save = tensor_to_image(img_u_save)
        img_target_save = tensor_to_image(img_target_save)
        img_u_save.save(os.path.join(image_dir, f"img_u_{str(i).zfill(5)}.png"))
        img_target_save.save(os.path.join(image_dir, f"img_target_{str(i).zfill(5)}.png"))
        generator.train()

    with torch.no_grad():
        generator.eval()
        if inversion is None: # for random
            for view, angle_y in enumerate(np.linspace(-angle_y_abs, angle_y_abs, sample_views)):
                cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                camera_params_view = torch.cat([cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
                img_u = generator.synthesis(w_u, camera_params_view)["image"]
                img_u = tensor_to_image(img_u)
                img_u.save(os.path.join(result_dir, f"unlearn_after_0_{view}.png"))
        else:
            if os.path.isdir(inversion_image_path): # for OOD
                for i in range(len(filenames)):
                    for view, angle_y in enumerate(np.linspace(-angle_y_abs, angle_y_abs, sample_views)):
                        cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                        camera_params_view = torch.cat([cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
                        img_origin = generator.synthesis(w_origin[[i]], camera_params_view)["image"]
                        img_origin = tensor_to_image(img_origin)
                        img_origin.save(os.path.join(result_dir, f"unlearn_after_{i}_{view}.png"))
            else: # for InD
                for view, angle_y in enumerate(np.linspace(-angle_y_abs, angle_y_abs, sample_views)):
                    cam2world_pose_view = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                    camera_params_view = torch.cat([cam2world_pose_view.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
                    img_u = generator.synthesis(w_u, camera_params_view)["image"]
                    img_u = tensor_to_image(img_u)
                    img_u.save(os.path.join(result_dir, f"unlearn_after_0_{view}.png"))
        generator.train()

    snapshot_data = dict()
    snapshot_data["G_ema"] = copy.deepcopy(generator).eval().requires_grad_(False).cpu()
    with open(os.path.join(ckpt_dir, f"last.pkl"), "wb") as f:
        pickle.dump(snapshot_data, f)
if __name__ == "__main__":
    unlearn() # pylint: disable=no-value-for-parameter