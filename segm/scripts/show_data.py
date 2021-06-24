import click
import numpy as np
from einops import rearrange
from PIL import Image
import torch

from segm.data.factory import create_dataset
from segm.data import utils
from segm.data.utils import IGNORE_LABEL

from toolbox.plotter import Plotter
from scipy.ndimage.morphology import distance_transform_edt


def visualize_batch(batch, normalization, dataset_kwargs):
    dataset_name = dataset_kwargs["dataset"]

    im0 = batch["im"]
    im0 = utils.rgb_denormalize(im0, normalization)
    im0 = rearrange(im0, "b c h w -> b h w c")
    im0 = (255 * im0.numpy()).astype(np.uint8)
    im_h, im_w = im0.shape[1], im0.shape[2]

    colors = batch["colors"]
    seg = batch["segmentation"]
    im1 = utils.seg_to_rgb(seg, colors)
    im1 = (255 * im1.numpy()).astype(np.uint8)

    alpha = 0.8
    for i in range(im0.shape[0]):
        im_a = Image.fromarray(im0[i])
        im_b = Image.fromarray(im1[i])
        im_blend = Image.blend(im_a, im_b, alpha).convert("RGB")
        im1[i] = np.asanyarray(im_blend)

    return im0, im1


def plot_batch_images(images, batch_size, sleep):
    n_images = len(images)
    plotter = Plotter(1, n_images)
    for i in range(batch_size):
        for j in range(n_images):
            if images[j][i] is not None:
                plotter.plot_im(images[j][i], 0, j)
        plotter.show(sleep)


def show(dataset_kwargs, sleep):
    batch_size = dataset_kwargs["batch_size"]

    print("load dataset")
    db = create_dataset(dataset_kwargs)
    print(f"length: {len(db.base_dataset)}")
    print("load batch")
    batch = next(iter(db))
    normalization = db.dataset.normalization

    processed_batch = visualize_batch(batch, normalization, dataset_kwargs)
    plot_batch_images(processed_batch, batch_size, sleep)


@click.command()
@click.argument("dataset_name", type=str)
@click.option("-imsz", "--im-size", default=0, type=int)
@click.option("-bs", "--batch-size", default=16, type=int)
@click.option("-nw", "--num-workers", default=4, type=int)
@click.option("-sl", "--sleep", default=3, type=float)
@click.option("-s", "--seed", default=0, type=int)
@click.option("-split", "--split", default="val", type=str)
def main(dataset_name, im_size, seed, batch_size, num_workers, sleep, split):
    torch.manual_seed(seed)
    np.random.seed(seed)
    normalization = "vit"
    dataset_kwargs = dict(
        dataset=dataset_name,
        image_size=im_size,
        batch_size=batch_size,
        num_workers=num_workers,
        split=split,
        normalization=normalization,
        crop=im_size > 0,
    )
    show(dataset_kwargs, sleep)


if __name__ == "__main__":
    main()
