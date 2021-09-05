import click
import einops
import torch
import torchvision

import matplotlib.pyplot as plt
import segm.utils.torch as ptu
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from segm import config
from segm.data.utils import STATS
from segm.model.decoder import MaskTransformer
from segm.model.factory import load_model
from torchvision import transforms


@click.command()
@click.argument("model-path", type=str)
@click.argument("image-path", type=str)
@click.argument("output-dir", type=str)
@click.option("--layer-id", default=0, type=int)
@click.option("--x-patch", default=0, type=int)
@click.option("--y-patch", default=0, type=int)
@click.option("--cmap", default="viridis", type=str)
@click.option("--enc/--dec", default=True, is_flag=True)
@click.option("--cls/--patch", default=False, is_flag=True)
def visualize(
    model_path,
    image_path,
    output_dir,
    layer_id,
    x_patch,
    y_patch,
    cmap,
    enc,
    cls,
):

    output_dir = Path(output_dir)
    model_dir = Path(model_path).parent

    ptu.set_gpu_mode(True)

    # Build model
    model, variant = load_model(model_path)
    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    model.to(ptu.device)

    # Get model config
    patch_size = model.patch_size
    normalization = variant["dataset_kwargs"]["normalization"]
    image_size = variant["dataset_kwargs"]["image_size"]
    n_cls = variant["net_kwargs"]["n_cls"]
    stats = STATS[normalization]

    # Open image and process it
    try:
        with open(image_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
    except:
        raise ValueError(f"Provided image path {image_path} is not a valid image file.")

    # Normalize and resize
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"]),
        ]
    )

    img = transform(img)

    # Make the image divisible by the patch size
    w, h = (
        image_size - image_size % patch_size,
        image_size - image_size % patch_size,
    )

    # Crop to image size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    # Sanity checks
    if not enc and not isinstance(model.decoder, MaskTransformer):
        raise ValueError(
            f"Attention maps for decoder are only availabe for MaskTransformer. Provided model with decoder type: {model.decoder}."
        )

    if not cls:
        if x_patch > w_featmap or y_patch > h_featmap:
            raise ValueError(
                f"Provided patch x: {x_patch} y: {y_patch} is not valid. Patch should be in the range x: [0, {w_featmap}), y: [0, {h_featmap})"
            )
        num_patch = w_featmap * y_patch + x_patch

    if layer_id < 0:
        raise ValueError("Provided layer_id should be positive.")

    if enc and model.encoder.n_layers <= layer_id:
        raise ValueError(
            f"Provided layer_id: {layer_id} is not valid for encoder with {model.encoder.n_layers}."
        )

    if not enc and model.decoder.n_layers <= layer_id:
        raise ValueError(
            f"Provided layer_id: {layer_id} is not valid for decoder with {model.decoder.n_layers}."
        )

    Path.mkdir(output_dir, exist_ok=True)

    # Process input and extract attention maps
    if enc:
        print(f"Generating Attention Mapping for Encoder Layer Id {layer_id}")
        attentions = model.get_attention_map_enc(img.to(ptu.device), layer_id)
        num_extra_tokens = 1 + model.encoder.distilled
        if cls:
            attentions = attentions[0, :, 0, num_extra_tokens:]
        else:
            attentions = attentions[
                0, :, num_patch + num_extra_tokens, num_extra_tokens:
            ]
    else:
        print(f"Generating Attention Mapping for Decoder Layer Id {layer_id}")
        attentions = model.get_attention_map_dec(img.to(ptu.device), layer_id)
        if cls:
            attentions = attentions[0, :, -n_cls:, :-n_cls]
        else:
            attentions = attentions[0, :, num_patch, :-n_cls]

    # Reshape into image shape
    nh = attentions.shape[0]  # Number of heads
    attentions = attentions.reshape(nh, -1)

    if cls and not enc:
        attentions = attentions.reshape(nh, n_cls, w_featmap, h_featmap)
    else:
        attentions = attentions.reshape(nh, 1, w_featmap, h_featmap)

    # Resize attention maps to match input size
    attentions = (
        F.interpolate(attentions, scale_factor=patch_size, mode="nearest").cpu().numpy()
    )

    # Save Attention map for each head
    for i in range(nh):
        base_name = "enc" if enc else "dec"
        head_name = f"{base_name}_layer{layer_id}_attn-head{i}"
        attention_maps_list = attentions[i]
        for j in range(attention_maps_list.shape[0]):
            attention_map = attention_maps_list[j]
            file_name = head_name
            dir_path = output_dir / f"{base_name}_layer{layer_id}"
            Path.mkdir(dir_path, exist_ok=True)
            if cls:
                if enc:
                    file_name = f"{file_name}_cls"
                    dir_path /= "cls"
                else:
                    file_name = f"{file_name}_{j}"
                    dir_path /= f"cls_{j}"
                Path.mkdir(dir_path, exist_ok=True)
            else:
                dir_path /= f"patch_{x_patch}_{y_patch}"
                Path.mkdir(dir_path, exist_ok=True)

            file_path = dir_path / f"{file_name}.png"
            plt.imsave(fname=str(file_path), arr=attention_map, format="png", cmap=cmap)
            print(f"{file_path} saved.")

    # Save input image showing selected patch
    if not cls:
        im_n = torchvision.utils.make_grid(img, normalize=True, scale_each=True)

        # Compute corresponding X and Y px in the original image
        x_px = x_patch * patch_size
        y_px = y_patch * patch_size
        px_v = einops.repeat(
            torch.tensor([1, 0, 0]),
            "c -> 1 c h w",
            h=patch_size,
            w=patch_size,
        )

        # Draw pixels for selected patch
        im_n[:, y_px : y_px + patch_size, x_px : x_px + patch_size] = px_v
        torchvision.utils.save_image(
            im_n,
            str(dir_path / "input_img.png"),
        )


if __name__ == "__main__":
    visualize()
