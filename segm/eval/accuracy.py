import click
import torch

import segm.utils.torch as ptu

from segm.utils.logger import MetricLogger

from segm.model.factory import create_vit
from segm.data.factory import create_dataset
from segm.data.utils import STATS
from segm.metrics import accuracy
from segm import config


def compute_labels(model, batch):
    im = batch["im"]
    target = batch["target"]

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            output = model.forward(im)
    acc1, acc5 = accuracy(output, target, topk=(1, 5))

    return acc1.item(), acc5.item()


def eval_dataset(model, dataset_kwargs):
    db = create_dataset(dataset_kwargs)
    print_freq = 20
    header = ""
    logger = MetricLogger(delimiter="  ")

    for batch in logger.log_every(db, print_freq, header):
        for k, v in batch.items():
            batch[k] = v.to(ptu.device)
        acc1, acc5 = compute_labels(model, batch)
        batch_size = batch["im"].size(0)
        logger.update(acc1=acc1, n=batch_size)
        logger.update(acc5=acc5, n=batch_size)
    print(f"Imagenet accuracy: {logger}")


@click.command()
@click.argument("backbone", type=str)
@click.option("--imagenet-dir", type=str)
@click.option("-bs", "--batch-size", default=32, type=int)
@click.option("-nw", "--num-workers", default=10, type=int)
@click.option("-gpu", "--gpu/--no-gpu", default=True, is_flag=True)
def main(backbone, imagenet_dir, batch_size, num_workers, gpu):
    ptu.set_gpu_mode(gpu)
    cfg = config.load_config()
    cfg = cfg["model"][backbone]
    cfg["backbone"] = backbone
    cfg["image_size"] = (cfg["image_size"], cfg["image_size"])

    dataset_kwargs = dict(
        dataset="imagenet",
        root_dir=imagenet_dir,
        image_size=cfg["image_size"],
        crop_size=cfg["image_size"],
        patch_size=cfg["patch_size"],
        batch_size=batch_size,
        num_workers=num_workers,
        split="val",
        normalization=STATS[cfg["normalization"]],
    )

    model = create_vit(cfg)
    model.to(ptu.device)
    model.eval()
    eval_dataset(model, dataset_kwargs)


if __name__ == "__main__":
    main()
