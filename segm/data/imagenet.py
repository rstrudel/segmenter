import os
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from PIL import Image

from segm.data import utils
from segm.config import dataset_dir


class ImagenetDataset(Dataset):
    def __init__(
        self,
        root_dir,
        image_size=224,
        crop_size=224,
        split="train",
        normalization="vit",
    ):
        super().__init__()
        assert image_size[0] == image_size[1]

        self.path = Path(root_dir) / split
        self.crop_size = crop_size
        self.image_size = image_size
        self.split = split
        self.normalization = normalization

        if split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.crop_size, interpolation=3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(image_size[0] + 32, interpolation=3),
                    transforms.CenterCrop(self.crop_size),
                    transforms.ToTensor(),
                ]
            )

        self.base_dataset = datasets.ImageFolder(self.path, self.transform)
        self.n_cls = 1000

    @property
    def unwrapped(self):
        return self

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        im, target = self.base_dataset[idx]
        im = utils.rgb_normalize(im, self.normalization)
        return dict(im=im, target=target)
