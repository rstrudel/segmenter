import torch
import torchvision.transforms.functional as F
import numpy as np
import yaml
from pathlib import Path

IGNORE_LABEL = 255
STATS = {
    "vit": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "deit": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
}


def seg_to_rgb(seg, colors):
    im = torch.zeros((seg.shape[0], seg.shape[1], seg.shape[2], 3)).float()
    cls = torch.unique(seg)
    for cl in cls:
        color = colors[int(cl)]
        if len(color.shape) > 1:
            color = color[0]
        im[seg == cl] = color
    return im


def dataset_cat_description(path, cmap=None):
    desc = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    colors = {}
    names = []
    for i, cat in enumerate(desc):
        names.append(cat["name"])
        if "color" in cat:
            colors[cat["id"]] = torch.tensor(cat["color"]).float() / 255
        else:
            colors[cat["id"]] = torch.tensor(cmap[cat["id"]]).float()
    colors[IGNORE_LABEL] = torch.tensor([0.0, 0.0, 0.0]).float()
    return names, colors


def rgb_normalize(x, stats):
    """
    x : C x *
    x \in [0, 1]
    """
    return F.normalize(x, stats["mean"], stats["std"])


def rgb_denormalize(x, stats):
    """
    x : N x C x *
    x \in [-1, 1]
    """
    mean = torch.tensor(stats["mean"])
    std = torch.tensor(stats["std"])
    for i in range(3):
        x[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return x
