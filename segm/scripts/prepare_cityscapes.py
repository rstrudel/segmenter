"""Prepare Cityscapes dataset"""
import click
import os
import shutil
import mmcv
import zipfile

from pathlib import Path
from segm.utils.download import download

USERNAME = None
PASSWORD = None


def download_cityscapes(path, username, password, overwrite=False):
    _CITY_DOWNLOAD_URLS = [
        ("gtFine_trainvaltest.zip", "99f532cb1af174f5fcc4c5bc8feea8c66246ddbc"),
        ("leftImg8bit_trainvaltest.zip", "2c0b77ce9933cc635adda307fbba5566f5d9d404"),
    ]
    download_dir = path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    os.system(
        f"wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username={username}&password={password}&submit=Login' https://www.cityscapes-dataset.com/login/ -P {download_dir}"
    )

    if not (download_dir / "gtFine_trainvaltest.zip").is_file():
        os.system(
            f"wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1 -P {download_dir}"
        )

    if not (download_dir / "leftImg8bit_trainvaltest.zip").is_file():
        os.system(
            f"wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3 -P {download_dir}"
        )

    for filename, checksum in _CITY_DOWNLOAD_URLS:
        # extract
        with zipfile.ZipFile(str(download_dir / filename), "r") as zip_ref:
            zip_ref.extractall(path=path)
        print("Extracted", filename)


def install_cityscapes_api():
    os.system("pip install cityscapesscripts")
    try:
        import cityscapesscripts
    except Exception:
        print(
            "Installing Cityscapes API failed, please install it manually %s"
            % (repo_url)
        )


def convert_json_to_label(json_file):
    from cityscapesscripts.preparation.json2labelImg import json2labelImg

    label_file = json_file.replace("_polygons.json", "_labelTrainIds.png")
    json2labelImg(json_file, label_file, "trainIds")


@click.command(help="Initialize Cityscapes dataset.")
@click.argument("download_dir", type=str)
@click.option("--username", default=USERNAME, type=str)
@click.option("--password", default=PASSWORD, type=str)
@click.option("--nproc", default=10, type=int)
def main(
    download_dir,
    username,
    password,
    nproc,
):

    dataset_dir = Path(download_dir) / "cityscapes"

    if username is None or password is None:
        raise ValueError(
            "You must indicate your username and password either in the script variables or by passing options --username and --pasword."
        )

    download_cityscapes(dataset_dir, username, password, overwrite=False)

    install_cityscapes_api()

    gt_dir = dataset_dir / "gtFine"

    poly_files = []
    for poly in mmcv.scandir(str(gt_dir), "_polygons.json", recursive=True):
        poly_file = str(gt_dir / poly)
        poly_files.append(poly_file)
    mmcv.track_parallel_progress(convert_json_to_label, poly_files, nproc)

    split_names = ["train", "val", "test"]

    for split in split_names:
        filenames = []
        for poly in mmcv.scandir(str(gt_dir / split), "_polygons.json", recursive=True):
            filenames.append(poly.replace("_gtFine_polygons.json", ""))
        with open(str(dataset_dir / f"{split}.txt"), "w") as f:
            f.writelines(f + "\n" for f in filenames)


if __name__ == "__main__":
    main()
