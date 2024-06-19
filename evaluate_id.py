import os
import click

import numpy as np

from glob import glob
from id_sim import IDSimNet

@click.command()
@click.option("--target_idx", type=int, default=0)
@click.option("--num_views", type=int, default=11)
@click.option("--exp")
def main(exp, target_idx, num_views):
    idsim_fn = IDSimNet().to("cuda")
    idsim_fn.eval()
    
    images_before = sorted(glob(os.path.join("experiments", exp, "training", "results", "unlearn_before*.png")))
    images_after = sorted(glob(os.path.join("experiments", exp, "training", "results", "unlearn_after*.png")))
    assert len(images_before) == len(images_after)

    idsims_avg = []
    idsims = []
    idsims_others = []

    for idx, (img1_path, img2_path) in enumerate(zip(images_before, images_after)):
        idsim_val = idsim_fn(img1_path, img2_path).item()

        idsims_avg.append(idsim_val)
        if idx // num_views == target_idx:
            idsims.append(idsim_val)
        else:
            idsims_others.append(idsim_val)
    
    print("ID Sim_avg: {:.4f}".format(np.mean(idsims_avg)))
    print("ID Sim: {:.4f}".format(np.mean(idsims)))
    print("ID Sim_others: {:.4f}".format(np.mean(idsims_others)))


if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter