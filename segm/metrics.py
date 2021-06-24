import torch
import numpy as np
import torch.distributed as dist
import segm.utils.torch as ptu

import os
import pickle as pkl
from pathlib import Path
import tempfile
import shutil
from mmseg.core import mean_iou

"""
ImageNet classifcation accuracy
"""


def accuracy(output, target, topk=(1,)):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            correct_k /= batch_size
            res.append(correct_k)
        return res


"""
Segmentation mean IoU
based on collect_results_cpu
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/apis/test.py#L160-L200
"""


def gather_data(seg_pred, tmp_dir=None):
    """
    distributed data gathering
    prediction and ground truth are stored in a common tmp directory
    and loaded on the master node to compute metrics
    """
    if tmp_dir is None:
        tmpprefix = os.path.expandvars("$WORK/temp")
    else:
        tmpprefix = tmp_dir
    MAX_LEN = 512
    # 32 is whitespace
    dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device=ptu.device)
    if ptu.dist_rank == 0:
        tmpdir = tempfile.mkdtemp(prefix=tmpprefix)
        tmpdir = torch.tensor(
            bytearray(tmpdir.encode()), dtype=torch.uint8, device=ptu.device
        )
        dir_tensor[: len(tmpdir)] = tmpdir
    # broadcast tmpdir from 0 to to the other nodes
    dist.broadcast(dir_tensor, 0)
    tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    tmpdir = Path(tmpdir)
    """
    Save results in temp file and load them on main process
    """
    tmp_file = tmpdir / f"part_{ptu.dist_rank}.pkl"
    pkl.dump(seg_pred, open(tmp_file, "wb"))
    dist.barrier()
    seg_pred = {}
    if ptu.dist_rank == 0:
        for i in range(ptu.world_size):
            part_seg_pred = pkl.load(open(tmpdir / f"part_{i}.pkl", "rb"))
            seg_pred.update(part_seg_pred)
        shutil.rmtree(tmpdir)
    return seg_pred


def compute_metrics(
    seg_pred,
    seg_gt,
    n_cls,
    ignore_index=None,
    ret_cat_iou=False,
    tmp_dir=None,
    distributed=False,
):
    ret_metrics_mean = torch.zeros(3, dtype=float, device=ptu.device)
    if ptu.dist_rank == 0:
        list_seg_pred = []
        list_seg_gt = []
        keys = sorted(seg_pred.keys())
        for k in keys:
            list_seg_pred.append(np.asarray(seg_pred[k]))
            list_seg_gt.append(np.asarray(seg_gt[k]))
        ret_metrics = mean_iou(
            results=list_seg_pred,
            gt_seg_maps=list_seg_gt,
            num_classes=n_cls,
            ignore_index=ignore_index,
        )
        ret_metrics = [ret_metrics["aAcc"], ret_metrics["Acc"], ret_metrics["IoU"]]
        ret_metrics_mean = torch.tensor(
            [
                np.round(np.nanmean(ret_metric.astype(np.float)) * 100, 2)
                for ret_metric in ret_metrics
            ],
            dtype=float,
            device=ptu.device,
        )
        cat_iou = ret_metrics[2]
    # broadcast metrics from 0 to all nodes
    if distributed:
        dist.broadcast(ret_metrics_mean, 0)
    pix_acc, mean_acc, miou = ret_metrics_mean
    ret = dict(pixel_accuracy=pix_acc, mean_accuracy=mean_acc, mean_iou=miou)
    if ret_cat_iou and ptu.dist_rank == 0:
        ret["cat_iou"] = cat_iou
    return ret
