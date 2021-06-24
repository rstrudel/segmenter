"""Prepare ADE20K dataset"""
import click
import zipfile

from pathlib import Path
from segm.utils.download import download


def download_ade(path, overwrite=False):
    _AUG_DOWNLOAD_URLS = [
        (
            "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
            "219e1696abb36c8ba3a3afe7fb2f4b4606a897c7",
        ),
        (
            "http://data.csail.mit.edu/places/ADEchallenge/release_test.zip",
            "e05747892219d10e9243933371a497e905a4860c",
        ),
    ]
    download_dir = path / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    for url, checksum in _AUG_DOWNLOAD_URLS:
        filename = download(
            url, path=str(download_dir), overwrite=overwrite, sha1_hash=checksum
        )
        # extract
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(path=str(path))


@click.command(help="Initialize ADE20K dataset.")
@click.argument("download_dir", type=str)
def main(download_dir):
    dataset_dir = Path(download_dir) / "ade20k"
    download_ade(dataset_dir, overwrite=False)


if __name__ == "__main__":
    main()
