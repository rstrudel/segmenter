"""Prepare PASCAL Context dataset"""
import click
import shutil
import tarfile
import torch

from tqdm import tqdm
from pathlib import Path

from segm.utils.download import download


def download_pcontext(path, overwrite=False):
    _AUG_DOWNLOAD_URLS = [
        (
            "https://www.dropbox.com/s/wtdibo9lb2fur70/VOCtrainval_03-May-2010.tar?dl=1",
            "VOCtrainval_03-May-2010.tar",
            "bf9985e9f2b064752bf6bd654d89f017c76c395a",
        ),
        (
            "https://codalabuser.blob.core.windows.net/public/trainval_merged.json",
            "",
            "169325d9f7e9047537fedca7b04de4dddf10b881",
        ),
        (
            "https://hangzh.s3.amazonaws.com/encoding/data/pcontext/train.pth",
            "",
            "4bfb49e8c1cefe352df876c9b5434e655c9c1d07",
        ),
        (
            "https://hangzh.s3.amazonaws.com/encoding/data/pcontext/val.pth",
            "",
            "ebedc94247ec616c57b9a2df15091784826a7b0c",
        ),
    ]
    download_dir = path / "downloads"

    download_dir.mkdir(parents=True, exist_ok=True)

    for url, filename, checksum in _AUG_DOWNLOAD_URLS:
        filename = download(
            url,
            path=str(download_dir / filename),
            overwrite=overwrite,
            sha1_hash=checksum,
        )
        # extract
        if Path(filename).suffix == ".tar":
            with tarfile.open(filename) as tar:
                tar.extractall(path=str(path))
        else:
            shutil.move(
                filename,
                str(path / "VOCdevkit" / "VOC2010" / Path(filename).name),
            )


@click.command(help="Initialize PASCAL Context dataset.")
@click.argument("download_dir", type=str)
def main(download_dir):

    dataset_dir = Path(download_dir) / "pcontext"

    download_pcontext(dataset_dir, overwrite=False)

    devkit_path = dataset_dir / "VOCdevkit"
    out_dir = devkit_path / "VOC2010" / "SegmentationClassContext"
    imageset_dir = devkit_path / "VOC2010" / "ImageSets" / "SegmentationContext"

    out_dir.mkdir(parents=True, exist_ok=True)
    imageset_dir.mkdir(parents=True, exist_ok=True)

    train_torch_path = devkit_path / "VOC2010" / "train.pth"
    val_torch_path = devkit_path / "VOC2010" / "val.pth"

    train_dict = torch.load(str(train_torch_path))

    train_list = []
    for idx, label in tqdm(train_dict.items()):
        idx = str(idx)
        new_idx = idx[:4] + "_" + idx[4:]
        train_list.append(new_idx)
        label_path = out_dir / f"{new_idx}.png"
        label.save(str(label_path))

    with open(str(imageset_dir / "train.txt"), "w") as f:
        f.writelines(line + "\n" for line in sorted(train_list))

    val_dict = torch.load(str(val_torch_path))

    val_list = []
    for idx, label in tqdm(val_dict.items()):
        idx = str(idx)
        new_idx = idx[:4] + "_" + idx[4:]
        val_list.append(new_idx)
        label_path = out_dir / f"{new_idx}.png"
        label.save(str(label_path))

    with open(str(imageset_dir / "val.txt"), "w") as f:
        f.writelines(line + "\n" for line in sorted(val_list))


if __name__ == "__main__":
    main()
