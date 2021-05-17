# Segmenter - Transformer for Semantic Segmentation

![Figure 1 from paper](./overview.png)

[Segmenter: Transformer for Semantic Segmentation](https://arxiv.org/abs/2105.05633)
by Robin Strudel, Ricardo Garcia, Ivan Laptev and Cordelia Schmid. 

## Installation

The code and several trained models will be released soon.

## Video Segmentation

Segmentation maps of Seg-B-Mask/16 trained on [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) segmentation dataset and tested on [DAVIS](https://davischallenge.org/) video dataset.

<p align="middle">
  <img src="https://github.com/rstrudel/segmenter/blob/master/gifs/choreography.gif" width="350">
  <img src="https://github.com/rstrudel/segmenter/blob/master/gifs/city-ride.gif" width="350">
</p>
<p align="middle">
  <img src="https://github.com/rstrudel/segmenter/blob/master/gifs/car-competition.gif" width="350">
  <img src="https://github.com/rstrudel/segmenter/blob/master/gifs/breakdance-flare.gif" width="350">
</p>

## BibTex

```
@article{strudel2021,
  title={Segmenter: Transformer for Semantic Segmentation},
  author={Strudel, Robin and Garcia, Ricardo and Laptev, Ivan and Schmid, Cordelia},
  journal={arXiv preprint arXiv:2105.05633},
  year={2021}
}
```


## Credits

The Vision Transformer code is based on [timm](https://github.com/rwightman/pytorch-image-models) library and the semantic segmentation training and evaluation pipeline 
is using [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).
