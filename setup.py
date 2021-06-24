import shutil

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))


def read_requirements_file(filename):
    req_file_path = path.join(path.dirname(path.realpath(__file__)), filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name="segm",
    version="0.0.1",
    description="Segmenter: Transformer for Semantic Segmentation",
    packages=find_packages(),
    install_requires=read_requirements_file("requirements.txt"),
)
