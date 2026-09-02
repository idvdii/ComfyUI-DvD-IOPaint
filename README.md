# ComfyUI-DvD-IOPaint

[简体中文](README.zh-CN.md) · English · [Changelog](CHANGELOG.md)

An interactive object-removal node for ComfyUI, adapted from the erase-model
runtime in [IOPaint](https://github.com/Sanster/IOPaint).

## Demos

### Interactive eraser

![DvD IOPaint demo](assets/demo.gif)

### SAM click-to-mask workflow

![DvD IOPaint SAM demo](assets/demo-sam.gif)

### External mask removal workflow

![DvD IOPaint external mask demo](assets/demo-external-mask.gif)

### Automatic mask generation

![DvD IOPaint Mask Generator demo](assets/demo-mask-generator.gif)

## Features

- Paint the removal mask directly on the node canvas.
- Release the pointer to run automatically when `auto_run` is enabled.
- The processed image replaces the canvas base, ready for the next edit.
- Drag a local image onto the node to start a new editing session.
- Brush preview follows the pointer; hold `Alt` and use the mouse wheel to
  change brush size. Without `Alt`, the wheel keeps ComfyUI's normal zoom.
- Undo an unprocessed mask stroke or restore the previous completed removal.
- Hover or drag across the comparison preview to inspect before/after results.
- Add **DvD IOPaint Mask Generator** to create foreground masks with RemoveBG
  or the anime-specific ISNet model, then connect the mask to the eraser.
- Add **DvD IOPaint SAM Interactive Segmentation**: click an object in the
  node to generate a reusable mask. Left-click adds foreground points and
  right-click adds background points.
- The eraser accepts optional `IMAGE` and `MASK` sockets. External masks are
  combined with the mask painted on the node, so the existing workflow remains
  usable.
- Models download only when first selected, with MD5 verification.
- Erase and mask models are stored in separate functional subdirectories below
  `ComfyUI/models/iopaint`.
- No full IOPaint installation and no replacement of ComfyUI's Torch build.

## Installation

### ComfyUI Manager

Use **Install via Git URL** and enter:

```text
https://github.com/idvdii/ComfyUI-DvD-IOPaint.git
```

Restart ComfyUI after installation.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/idvdii/ComfyUI-DvD-IOPaint.git
cd ComfyUI-DvD-IOPaint
python -m pip install -r requirements.txt
# Install this optional file when using Mask Generator
python -m pip install -r requirements-mask.txt
```

Use the Python executable that belongs to your ComfyUI installation. Portable
Windows packages usually provide an embedded Python executable in the folder
next to `ComfyUI`.

If you only use the original eraser, `requirements-mask.txt` is not needed.
Mask Generator requires `rembg`; its selected ONNX model is downloaded on first
use and the node never installs or changes ComfyUI's Torch/CUDA packages while
executing.

## Usage

1. Add **DvD IOPaint Interactive Eraser** from `DvD/Image`.
2. Select or drop an image onto the node.
3. Select a model and paint the object or text to remove.
4. Release the pointer. With `auto_run` enabled, the workflow is queued
   automatically and the result becomes the new editable image.

Toolbar controls:

| Control | Action |
| --- | --- |
| Pencil | Paint the removal mask |
| Eraser | Remove parts of the current mask |
| Undo | Undo the last unprocessed mask stroke |
| History | Restore the image before the last completed removal |
| Trash | Clear the current mask |

The node outputs the processed `IMAGE` and the submitted `MASK`.

`edge_blur` softens the join between the repaired area and the source image;
the default `0` leaves the result unchanged. For painting directly in the
eraser, try `2–6` first. When `mask_input` is connected, values above `0`
soften the edge of the combined painted and external mask; `0` preserves the
external mask. `opencv_radius` only affects the two OpenCV models and is
ignored by every other model.

### SAM interactive mask

1. Add **DvD IOPaint SAM Interactive Segmentation** and connect an `IMAGE`,
   or choose an image in its file widget.
2. Select a SAM ViT model. Left-click the object to add a foreground point;
   right-click unwanted areas to add background points.
3. `auto_run` is disabled by default so multiple points can be placed first.
   Press the play button to submit, or enable `auto_run` to refresh after every click.
4. The upper `image` output is the unchanged source image. Connect the lower
   `mask` output to the eraser's `mask_input`.

`mask_expand` grows the final mask outward by the selected number of pixels
(`0` keeps the original boundary); `mask_blur` softens its edge.

SAM1 follows the clicked object's visual boundaries; it does not understand a
text category. More foreground/background points improve difficult or
occluded selections.

### Automatic mask generation

1. Add **DvD IOPaint Mask Generator** from `DvD/Image` and connect an `IMAGE`.
2. Select Anime Segmentation or a RemoveBG model. The selected weight is only
   downloaded on first use.
3. Connect its `mask` output to the eraser's `mask_input` socket.
4. Enable `invert` when you need a background mask instead of a foreground
   mask; `feather` softens the resulting edge.

The generator also returns a `foreground` preview. Its background is black by
default; enter `#RRGGBB`, `#RGB`, a common color name, or `R,G,B` in
`background_color` to choose a solid preview/output background. The output is
an RGB IMAGE without an alpha channel, so use the `mask` output as the alpha
when you need a transparent composite.

## Models

Download sizes are rounded so users can estimate disk and network use.

| Model | Download | Model files | Directory | Notes |
| --- | ---: | --- | --- | --- |
| LaMa | ~196 MiB | [big-lama.pt](https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt) | `erase` | General-purpose large-mask removal |
| Anime LaMa | ~196 MiB | [anime-manga-big-lama.pt](https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt) | `erase` | Anime and manga-oriented LaMa weights |
| AOT Manga/Anime | ~22 MiB | [aot_traced.pt](https://huggingface.co/ogkalu/aot-inpainting/resolve/42ffc84ff1bd46dd95f1c5a41e83ee7e98f39189/aot_traced.pt) | `erase` | Lightweight color anime/manga option |
| MAT | ~239 MiB | [Places_512_FullData_G.pth](https://github.com/Sanster/models/releases/download/add_mat/Places_512_FullData_G.pth) | `erase` | Structure-aware general inpainting |
| MIGAN | ~27 MiB | [migan_traced.pt](https://github.com/Sanster/models/releases/download/migan/migan_traced.pt) | `erase` | Lightweight 512-based inpainting |
| LDM | ~1.6 GiB, 3 files | [encode](https://github.com/Sanster/models/releases/download/add_ldm/cond_stage_model_encode.pt) · [decode](https://github.com/Sanster/models/releases/download/add_ldm/cond_stage_model_decode.pt) · [diffusion](https://github.com/Sanster/models/releases/download/add_ldm/diffusion.pt) | `erase` | Diffusion-based erase model; slower and heavy |
| ZITS | ~600 MiB, 4 files | [inpaint](https://github.com/Sanster/models/releases/download/add_zits/zits-inpaint-0717.pt) · [edge-line](https://github.com/Sanster/models/releases/download/add_zits/zits-edge-line-0717.pt) · [structure](https://github.com/Sanster/models/releases/download/add_zits/zits-structure-upsample-0717.pt) · [wireframe](https://github.com/Sanster/models/releases/download/add_zits/zits-wireframe-0717.pt) | `erase` | Structure and line-aware inpainting |
| FcF | ~327 MiB | [places_512_G.pth](https://github.com/Sanster/models/releases/download/add_fcf/places_512_G.pth) | `erase` | Fourier-based large-hole completion |
| Manga B&W Semantic | ~235 MiB, 2 files | [inpaintor](https://github.com/Sanster/models/releases/download/manga/manga_inpaintor.jit) · [line model](https://github.com/Sanster/models/releases/download/manga/erika.jit) | `erase` | Black-and-white manga only; masked output is grayscale |
| OpenCV Telea | 0 MiB | No download required | Not applicable | Fast classical inpainting for small defects |
| OpenCV Navier-Stokes | 0 MiB | No download required | Not applicable | Fast classical inpainting for small defects |

Mask Generator models:

| Model | Download (approx.) | Model file | Directory |
| --- | ---: | --- | --- |
| Anime Segmentation / ISNet Anime | 170 MiB | [isnet-anime.onnx](https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx) | `mask/anime_seg` |
| RemoveBG / U2Net | 176 MiB | [u2net.onnx](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx) | `mask/removebg` |
| RemoveBG / U2NetP | 4.7 MiB | [u2netp.onnx](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx) | `mask/removebg` |
| RemoveBG / ISNet General | 167 MiB | [isnet-general-use.onnx](https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx) | `mask/removebg` |
| RemoveBG / BiRefNet Lite | 90 MiB | [BiRefNet General Lite](https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx) | `mask/removebg` |
| RemoveBG / BiRefNet | 443 MiB | [BiRefNet General](https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx) | `mask/removebg` |
| RemoveBG / Silueta | 44 MiB | [silueta.onnx](https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx) | `mask/removebg` |

SAM interactive-segmentation models:

| Model | Download (approx.) | Model file | Directory |
| --- | ---: | --- | --- |
| SAM ViT-B | 375 MiB | [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) | `interactive_seg` |
| SAM ViT-L | 1.25 GiB | [sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) | `interactive_seg` |
| SAM ViT-H | 2.56 GiB | [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) | `interactive_seg` |

The first use of a neural model requires access to its GitHub or Hugging Face
download URL. Interrupted or invalid files are removed, then downloaded again
on the next run.

Automatic download is recommended. For a manual download, keep the filename
shown in the table and place it in the listed directory below
`ComfyUI/models/iopaint`; multi-file models require every linked file.

Model layout:

```text
ComfyUI/models/iopaint/
├─ erase/                 # LaMa, AOT, MAT, MIGAN and other erase models
├─ mask/
   ├─ anime_seg/          # Anime Segmentation / ISNet Anime
   └─ removebg/           # rembg U2Net, ISNet, BiRefNet and related models
└─ interactive_seg/       # SAM1 checkpoints (downloaded on first click)
```

For older installations, verified erase files found directly in
`models/iopaint` are migrated to `erase` the first time they are checked.
Unrelated files are left untouched.

## Dependencies and compatibility

`requirements.txt` contains only the extra packages needed by this node:

- `opencv-contrib-python-headless` for image processing and classical inpainting
- `scikit-image` for ZITS edge and line processing

Optional Mask Generator dependencies are listed in `requirements-mask.txt`:

- `rembg` for RemoveBG and ISNet Anime ONNX inference and first-use downloads.

SAM interactive segmentation is self-contained: the node vendors the small
SAM1 Python implementation and reuses ComfyUI's existing Torch/Numpy. No SAM
package, CUDA toolkit or Torch replacement is required. SAM checkpoints are
downloaded only after the first point is added and stored in
`models/iopaint/interactive_seg`.

This file does not pin `torch`, `torchvision` or CUDA. The default rembg
dependency uses CPU ONNX Runtime. If you already use a GPU ONNX Runtime build,
keep the version matching your installation and review pip's changes so it is
not replaced.

Torch, NumPy, Pillow, Pydantic, SciPy and tqdm are supplied by ComfyUI and are
intentionally not pinned here. This avoids replacing the user's CUDA-specific
Torch installation. The plugin vendors only the adapted erase-model runtime;
it does not install the IOPaint application package.

Tested with ComfyUI `v0.30.2`, Python 3.13, PyTorch 2.9.1 and CUDA. CPU
inference also works but is slower. The included smoke test can be run from the
ComfyUI directory:

```bash
python custom_nodes/ComfyUI-DvD-IOPaint/tests/test_smoke.py
```

## Credits and license

Erase-model URLs, checksums, preprocessing behavior and adapted runtime sources
are based on IOPaint 1.6.0. See [NOTICE](NOTICE) for the exact upstream commit
and modification notes.

Licensed under the [Apache License 2.0](LICENSE).
