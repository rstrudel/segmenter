import sys
import click
from pathlib import Path
import yaml
import numpy as np
from PIL import Image
import shutil

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from segm.utils import distributed
from segm.utils.logger import MetricLogger
import segm.utils.torch as ptu

from segm.model.factory import load_model
from segm.data.factory import create_dataset
from segm.metrics import gather_data, compute_metrics

from segm.model.utils import inference
from segm.data.utils import seg_to_rgb, rgb_denormalize, IGNORE_LABEL
from segm import config


def blend_im(im, seg, alpha=0.5):
    pil_im = Image.fromarray(im)
    pil_seg = Image.fromarray(seg)
    im_blend = Image.blend(pil_im, pil_seg, alpha).convert("RGB")
    return np.asarray(im_blend)


def save_im(save_dir, save_name, im, seg_pred, seg_gt, colors, blend, normalization):
    seg_rgb = seg_to_rgb(seg_gt[None], colors)
    pred_rgb = seg_to_rgb(seg_pred[None], colors)
    im_unnorm = rgb_denormalize(im, normalization)
    save_dir = Path(save_dir)

    # save images
    im_uint = (im_unnorm.permute(0, 2, 3, 1).cpu().numpy()).astype(np.uint8)
    seg_rgb_uint = (255 * seg_rgb.cpu().numpy()).astype(np.uint8)
    seg_pred_uint = (255 * pred_rgb.cpu().numpy()).astype(np.uint8)
    for i in range(pred_rgb.shape[0]):
        if blend:
            blend_pred = blend_im(im_uint[i], seg_pred_uint[i])
            blend_gt = blend_im(im_uint[i], seg_rgb_uint[i])
            ims = (im_uint[i], blend_pred, blend_gt)
        else:
            ims = (im_uint[i], seg_pred_uint[i], seg_rgb_uint[i])
        for im, im_dir in zip(
            ims,
            (save_dir / "input", save_dir / "pred", save_dir / "gt"),
        ):
            pil_out = Image.fromarray(im)
            im_dir.mkdir(exist_ok=True)
            pil_out.save(im_dir / save_name)


def process_batch(
    model,
    batch,
    window_size,
    window_stride,
    window_batch_size,
):
    ims = batch["im"]
    ims_metas = batch["im_metas"]
    ori_shape = ims_metas[0]["ori_shape"]
    ori_shape = (ori_shape[0].item(), ori_shape[1].item())
    filename = batch["im_metas"][0]["ori_filename"][0]

    model_without_ddp = model
    if ptu.distributed:
        model_without_ddp = model.module
    seg_pred = inference(
        model_without_ddp,
        ims,
        ims_metas,
        ori_shape,
        window_size,
        window_stride,
        window_batch_size,
    )
    seg_pred = seg_pred.argmax(0)
    im = F.interpolate(ims[-1], ori_shape, mode="bilinear")

    return filename, im.cpu(), seg_pred.cpu()


def eval_dataset(
    model,
    multiscale,
    model_dir,
    blend,
    window_size,
    window_stride,
    window_batch_size,
    save_images,
    frac_dataset,
    dataset_kwargs,
):
    db = create_dataset(dataset_kwargs)
    normalization = db.dataset.normalization
    dataset_name = dataset_kwargs["dataset"]
    im_size = dataset_kwargs["image_size"]
    cat_names = db.base_dataset.names
    n_cls = db.unwrapped.n_cls
    if multiscale:
        db.dataset.set_multiscale_mode()

    logger = MetricLogger(delimiter="  ")
    header = ""
    print_freq = 50

    ims = {}
    seg_pred_maps = {}
    idx = 0
    for batch in logger.log_every(db, print_freq, header):
        colors = batch["colors"]
        filename, im, seg_pred = process_batch(
            model,
            batch,
            window_size,
            window_stride,
            window_batch_size,
        )
        ims[filename] = im
        seg_pred_maps[filename] = seg_pred
        idx += 1
        if idx > len(db) * frac_dataset:
            break

    seg_gt_maps = db.dataset.get_gt_seg_maps()
    if save_images:
        save_dir = model_dir / "images"
        if ptu.dist_rank == 0:
            if save_dir.exists():
                shutil.rmtree(save_dir)
            save_dir.mkdir()
        if ptu.distributed:
            torch.distributed.barrier()

        for name in sorted(ims):
            instance_dir = save_dir
            filename = name

            if dataset_name == "cityscapes":
                filename_list = name.split("/")
                instance_dir = instance_dir / filename_list[0]
                filename = filename_list[-1]
                if not instance_dir.exists():
                    instance_dir.mkdir()

            save_im(
                instance_dir,
                filename,
                ims[name],
                seg_pred_maps[name],
                torch.tensor(seg_gt_maps[name]),
                colors,
                blend,
                normalization,
            )
        if ptu.dist_rank == 0:
            shutil.make_archive(save_dir, "zip", save_dir)
            # shutil.rmtree(save_dir)
            print(f"Saved eval images in {save_dir}.zip")

    if ptu.distributed:
        torch.distributed.barrier()
        seg_pred_maps = gather_data(seg_pred_maps)

    scores = compute_metrics(
        seg_pred_maps,
        seg_gt_maps,
        n_cls,
        ignore_index=IGNORE_LABEL,
        ret_cat_iou=True,
        distributed=ptu.distributed,
    )

    if ptu.dist_rank == 0:
        scores["inference"] = "single_scale" if not multiscale else "multi_scale"
        suffix = "ss" if not multiscale else "ms"
        scores["cat_iou"] = np.round(100 * scores["cat_iou"], 2).tolist()
        for k, v in scores.items():
            if k != "cat_iou" and k != "inference":
                scores[k] = v.item()
            if k != "cat_iou":
                print(f"{k}: {scores[k]}")
        scores_str = yaml.dump(scores)
        with open(model_dir / f"scores_{suffix}.yml", "w") as f:
            f.write(scores_str)


@click.command()
@click.argument("model_path", type=str)
@click.argument("dataset_name", type=str)
@click.option("--im-size", default=None, type=int)
@click.option("--multiscale/--singlescale", default=False, is_flag=True)
@click.option("--blend/--no-blend", default=True, is_flag=True)
@click.option("--window-size", default=None, type=int)
@click.option("--window-stride", default=None, type=int)
@click.option("--window-batch-size", default=4, type=int)
@click.option("--save-images/--no-save-images", default=False, is_flag=True)
@click.option("-frac-dataset", "--frac-dataset", default=1.0, type=float)
def main(
    model_path,
    dataset_name,
    im_size,
    multiscale,
    blend,
    window_size,
    window_stride,
    window_batch_size,
    save_images,
    frac_dataset,
):

    model_dir = Path(model_path).parent

    # start distributed mode
    ptu.set_gpu_mode(True)
    distributed.init_process()

    model, variant = load_model(model_path)
    patch_size = model.patch_size
    model.eval()
    model.to(ptu.device)
    if ptu.distributed:
        model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)

    cfg = config.load_config()
    dataset_cfg = cfg["dataset"][dataset_name]
    normalization = variant["dataset_kwargs"]["normalization"]
    if im_size is None:
        im_size = dataset_cfg.get("im_size", variant["dataset_kwargs"]["image_size"])
    if window_size is None:
        window_size = dataset_cfg.get(
            "window_size", variant["dataset_kwargs"]["crop_size"]
        )
    if window_stride is None:
        window_stride = dataset_cfg.get(
            "window_stride", variant["dataset_kwargs"]["crop_size"]
        )

    dataset_kwargs = dict(
        dataset=dataset_name,
        image_size=im_size,
        crop_size=im_size,
        patch_size=patch_size,
        batch_size=1,
        num_workers=10,
        split="val",
        normalization=normalization,
        crop=False,
        rep_aug=False,
    )

    eval_dataset(
        model,
        multiscale,
        model_dir,
        blend,
        window_size,
        window_stride,
        window_batch_size,
        save_images,
        frac_dataset,
        dataset_kwargs,
    )

    distributed.barrier()
    distributed.destroy_process()
    sys.exit(1)


if __name__ == "__main__":
    main()
