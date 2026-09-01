# ComfyUI-DvD-IOPaint

An interactive object-removal node for ComfyUI, adapted from the erase-model
runtime in [IOPaint](https://github.com/Sanster/IOPaint).

在一个 ComfyUI 节点内完成图片载入、遮罩涂抹、自动消除、前后对比和连续编辑。

![DvD IOPaint demo](assets/demo.gif)

## Features

- Paint the removal mask directly on the node canvas.
- Release the pointer to run automatically when `auto_run` is enabled.
- The processed image replaces the canvas base, ready for the next edit.
- Drag a local image onto the node to start a new editing session.
- Brush preview follows the pointer; hold `Alt` and use the mouse wheel to
  change brush size. Without `Alt`, the wheel keeps ComfyUI's normal zoom.
- Undo an unprocessed mask stroke or restore the previous completed removal.
- Hover or drag across the comparison preview to inspect before/after results.
- Models download only when first selected, with MD5 verification.
- Model files are stored in `ComfyUI/models/iopaint`.
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
```

Use the Python executable that belongs to your ComfyUI installation. Portable
Windows packages usually provide an embedded Python executable in the folder
next to `ComfyUI`.

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

## Models

Download sizes are rounded so users can estimate disk and network use.

| Model | Download | Notes |
| --- | ---: | --- |
| LaMa | ~196 MiB | General-purpose large-mask removal |
| Anime LaMa | ~196 MiB | Anime and manga-oriented LaMa weights |
| AOT Manga/Anime | ~22 MiB | Lightweight color anime/manga option |
| MAT | ~239 MiB | Structure-aware general inpainting |
| MIGAN | ~27 MiB | Lightweight 512-based inpainting |
| LDM | ~1.6 GiB, 3 files | Diffusion-based erase model; slower and heavy |
| ZITS | ~600 MiB, 4 files | Structure and line-aware inpainting |
| FcF | ~327 MiB | Fourier-based large-hole completion |
| Manga B&W Semantic | ~235 MiB, 2 files | Black-and-white manga only; masked output is grayscale |
| OpenCV Telea | 0 MiB | Fast classical inpainting for small defects |
| OpenCV Navier-Stokes | 0 MiB | Fast classical inpainting for small defects |

The first use of a neural model requires access to its GitHub or Hugging Face
download URL. Interrupted or invalid files are removed, then downloaded again
on the next run.

## Dependencies and compatibility

`requirements.txt` contains only the extra packages needed by this node:

- `opencv-contrib-python-headless` for image processing and classical inpainting
- `scikit-image` for ZITS edge and line processing

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

## 中文说明

- 开启 `auto_run` 后，画完松开鼠标会自动运行，结果直接覆盖操作区底图。
- 将图片拖到节点上即可替换输入图片并开始新的编辑。
- 鼠标位于画布时会显示半透明画笔直径；按住 `Alt` 滚动滚轮可调节大小。
- 不按 `Alt` 时保留 ComfyUI 原本的画布缩放操作。
- 工具栏的普通撤销用于撤销尚未处理的笔画，历史按钮用于返回上一次已完成的消除结果。
- 下方对比区左侧为处理前、右侧为处理后，可移动鼠标查看分界。
- 模型首次使用时自动下载到 `ComfyUI/models/iopaint`，之后直接使用本地文件。
- `Manga B&W Semantic` 仅推荐黑白漫画；彩色二次元图片优先尝试 AOT 或 Anime LaMa。

## Credits and license

Erase-model URLs, checksums, preprocessing behavior and adapted runtime sources
are based on IOPaint 1.6.0. See [NOTICE](NOTICE) for the exact upstream commit
and modification notes.

Licensed under the [Apache License 2.0](LICENSE).
