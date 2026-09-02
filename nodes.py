import hashlib
import importlib
import io
import json
import logging
import math
import os
import sys
import threading
from contextlib import contextmanager

import cv2
import numpy as np
import torch
from PIL import Image, ImageColor, ImageOps
from torch.hub import download_url_to_file

import comfy.model_management as model_management
import folder_paths
import node_helpers
from nodes import PreviewImage


LOGGER = logging.getLogger(__name__)
MODEL_ROOT_DIRECTORY = os.path.join(folder_paths.models_dir, "iopaint")
ERASE_MODEL_DIRECTORY = os.path.join(MODEL_ROOT_DIRECTORY, "erase")
MASK_MODEL_DIRECTORY = os.path.join(MODEL_ROOT_DIRECTORY, "mask")
MASK_REMBG_DIRECTORY = os.path.join(MASK_MODEL_DIRECTORY, "removebg")
MASK_ANIME_DIRECTORY = os.path.join(MASK_MODEL_DIRECTORY, "anime_seg")
INTERACTIVE_SEG_DIRECTORY = os.path.join(MODEL_ROOT_DIRECTORY, "interactive_seg")
# Keep the old name as a compatibility hook for existing callers and tests.
MODEL_DIRECTORY = ERASE_MODEL_DIRECTORY
PLUGIN_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
if PLUGIN_DIRECTORY not in sys.path:
    sys.path.insert(0, PLUGIN_DIRECTORY)
# The vendored IOPaint erase runtime reads this variable when resolving its
# model URLs.  Mask backends use their own scoped environment below.
os.environ["DVD_IOPAINT_MODEL_DIR"] = ERASE_MODEL_DIRECTORY

MODEL_SPECS = {
    "lama": {
        "display": "LaMa (~196 MiB)",
        "module": "lama",
        "class": "LaMa",
        "files": (
            {
                "filename": "big-lama.pt",
                "url": "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
                "md5": "e3aa4aaa15225a33ec84f9f4bc47e500",
            },
        ),
    },
    "anime_lama": {
        "display": "Anime LaMa (~196 MiB)",
        "module": "lama",
        "class": "AnimeLaMa",
        "files": (
            {
                "filename": "anime-manga-big-lama.pt",
                "url": "https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt",
                "md5": "29f284f36a0a510bcacf39ecf4c4d54f",
            },
        ),
    },
    "aot": {
        "display": "AOT Manga/Anime (~22 MiB)",
        "module": "aot",
        "class": "AOT",
        "files": (
            {
                "filename": "aot_traced.pt",
                "url": "https://huggingface.co/ogkalu/aot-inpainting/resolve/"
                "42ffc84ff1bd46dd95f1c5a41e83ee7e98f39189/aot_traced.pt",
                "md5": "82350e9ce753fbdf59bc9250bcf4dd73",
            },
        ),
    },
    "mat": {
        "display": "MAT (~239 MiB)",
        "module": "mat",
        "class": "MAT",
        "files": (
            {
                "filename": "Places_512_FullData_G.pth",
                "url": "https://github.com/Sanster/models/releases/download/add_mat/Places_512_FullData_G.pth",
                "md5": "8ca927835fa3f5e21d65ffcb165377ed",
            },
        ),
    },
    "migan": {
        "display": "MIGAN (~27 MiB)",
        "module": "mi_gan",
        "class": "MIGAN",
        "files": (
            {
                "filename": "migan_traced.pt",
                "url": "https://github.com/Sanster/models/releases/download/migan/migan_traced.pt",
                "md5": "76eb3b1a71c400ee3290524f7a11b89c",
            },
        ),
    },
    "ldm": {
        "display": "LDM (~1.6 GiB, 3 files)",
        "module": "ldm",
        "class": "LDM",
        "files": (
            {
                "filename": "cond_stage_model_encode.pt",
                "url": "https://github.com/Sanster/models/releases/download/add_ldm/cond_stage_model_encode.pt",
                "md5": "23239fc9081956a3e70de56472b3f296",
            },
            {
                "filename": "cond_stage_model_decode.pt",
                "url": "https://github.com/Sanster/models/releases/download/add_ldm/cond_stage_model_decode.pt",
                "md5": "fe419cd15a750d37a4733589d0d3585c",
            },
            {
                "filename": "diffusion.pt",
                "url": "https://github.com/Sanster/models/releases/download/add_ldm/diffusion.pt",
                "md5": "b0afda12bf790c03aba2a7431f11d22d",
            },
        ),
    },
    "zits": {
        "display": "ZITS (~600 MiB, 4 files)",
        "module": "zits",
        "class": "ZITS",
        "files": (
            {
                "filename": "zits-inpaint-0717.pt",
                "url": "https://github.com/Sanster/models/releases/download/add_zits/zits-inpaint-0717.pt",
                "md5": "9978cc7157dc29699e42308d675b2154",
            },
            {
                "filename": "zits-edge-line-0717.pt",
                "url": "https://github.com/Sanster/models/releases/download/add_zits/zits-edge-line-0717.pt",
                "md5": "55e31af21ba96bbf0c80603c76ea8c5f",
            },
            {
                "filename": "zits-structure-upsample-0717.pt",
                "url": "https://github.com/Sanster/models/releases/download/add_zits/zits-structure-upsample-0717.pt",
                "md5": "3d88a07211bd41b2ec8cc0d999f29927",
            },
            {
                "filename": "zits-wireframe-0717.pt",
                "url": "https://github.com/Sanster/models/releases/download/add_zits/zits-wireframe-0717.pt",
                "md5": "a9727c63a8b48b65c905d351b21ce46b",
            },
        ),
    },
    "fcf": {
        "display": "FcF (~327 MiB)",
        "module": "fcf",
        "class": "FcF",
        "files": (
            {
                "filename": "places_512_G.pth",
                "url": "https://github.com/Sanster/models/releases/download/add_fcf/places_512_G.pth",
                "md5": "3323152bc01bf1c56fd8aba74435a211",
            },
        ),
    },
    "manga": {
        "display": "Manga B&W Semantic (~235 MiB, 2 files)",
        "module": "manga",
        "class": "Manga",
        "files": (
            {
                "filename": "manga_inpaintor.jit",
                "url": "https://github.com/Sanster/models/releases/download/manga/manga_inpaintor.jit",
                "md5": "7d8b269c4613b6b3768af714610da86c",
            },
            {
                "filename": "erika.jit",
                "url": "https://github.com/Sanster/models/releases/download/manga/erika.jit",
                "md5": "0c926d5a4af8450b0d00bc5b9a095644",
            },
        ),
    },
}
OPENCV_CHOICES = {
    "opencv_telea": "OpenCV Telea (0 MiB)",
    "opencv_ns": "OpenCV Navier-Stokes (0 MiB)",
}
MODEL_CHOICES = tuple(
    spec["display"] for spec in MODEL_SPECS.values()
) + tuple(OPENCV_CHOICES.values())
DEFAULT_MODEL = MODEL_SPECS["lama"]["display"]
MODEL_ALIASES = {
    **{spec["display"]: key for key, spec in MODEL_SPECS.items()},
    **{display: key for key, display in OPENCV_CHOICES.items()},
    "LaMa": "lama",
    "Anime LaMa": "anime_lama",
    "AOT": "aot",
    "AOT Manga/Anime": "aot",
    "MAT": "mat",
    "MIGAN": "migan",
    "LDM": "ldm",
    "ZITS": "zits",
    "FcF": "fcf",
    "Manga": "manga",
    "Manga (~80 MiB, 2 files)": "manga",
    "Manga B&W Semantic": "manga",
    "OpenCV Telea": "opencv_telea",
    "OpenCV Navier-Stokes": "opencv_ns",
}

# Phase 1 mask plugins.  The RemoveBG entries use rembg's maintained session
# implementations, while Anime Segmentation uses its anime-specific ISNet
# session.  The display strings intentionally include approximate download
# sizes so a user can choose a lightweight model before starting a run.
MASK_MODEL_SPECS = {
    "anime_seg": {
        "display": "Anime Segmentation / ISNet Anime (~170 MiB)",
        "rembg_model": "isnet-anime",
        "directory": MASK_ANIME_DIRECTORY,
    },
    "removebg_u2net": {
        "display": "RemoveBG / U2Net (~176 MiB)",
        "rembg_model": "u2net",
        "directory": MASK_REMBG_DIRECTORY,
    },
    "removebg_u2netp": {
        "display": "RemoveBG / U2NetP (~4.7 MiB)",
        "rembg_model": "u2netp",
        "directory": MASK_REMBG_DIRECTORY,
    },
    "removebg_isnet": {
        "display": "RemoveBG / ISNet General (~167 MiB)",
        "rembg_model": "isnet-general-use",
        "directory": MASK_REMBG_DIRECTORY,
    },
    "removebg_birefnet_lite": {
        "display": "RemoveBG / BiRefNet Lite (~90 MiB)",
        "rembg_model": "birefnet-general-lite",
        "directory": MASK_REMBG_DIRECTORY,
    },
    "removebg_birefnet": {
        "display": "RemoveBG / BiRefNet (~443 MiB)",
        "rembg_model": "birefnet-general",
        "directory": MASK_REMBG_DIRECTORY,
    },
    "removebg_silueta": {
        "display": "RemoveBG / Silueta (~44 MiB)",
        "rembg_model": "silueta",
        "directory": MASK_REMBG_DIRECTORY,
    },
}
MASK_MODEL_CHOICES = tuple(
    spec["display"] for spec in MASK_MODEL_SPECS.values()
)
MASK_MODEL_ALIASES = {
    **{spec["display"]: key for key, spec in MASK_MODEL_SPECS.items()},
    "Anime Segmentation": "anime_seg",
    "ISNet Anime": "anime_seg",
    "RemoveBG": "removebg_u2net",
    "U2Net": "removebg_u2net",
    "U2NetP": "removebg_u2netp",
    "ISNet General": "removebg_isnet",
    "BiRefNet Lite": "removebg_birefnet_lite",
    "BiRefNet": "removebg_birefnet",
    "Silueta": "removebg_silueta",
}

# Phase 2 interactive segmentation models.  The SAM implementation is
# vendored below ``dvd_iopaint_vendor`` so users do not need to install the
# upstream package or change ComfyUI's Torch/CUDA environment.  Checkpoints
# remain outside the repository and are downloaded only after the first click
# in the interactive node.
SAM_MODEL_SPECS = {
    "vit_b": {
        "display": "SAM ViT-B (~375 MiB)",
        "architecture": "vit_b",
        "files": (
            {
                "filename": "sam_vit_b_01ec64.pth",
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                "md5": "01ec64d29a2fca3f0661936605ae66f8",
            },
        ),
    },
    "vit_l": {
        "display": "SAM ViT-L (~1.25 GiB)",
        "architecture": "vit_l",
        "files": (
            {
                "filename": "sam_vit_l_0b3195.pth",
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                "md5": "0b3195507c641ddb6910d2bb5adee89c",
            },
        ),
    },
    "vit_h": {
        "display": "SAM ViT-H (~2.56 GiB)",
        "architecture": "vit_h",
        "files": (
            {
                "filename": "sam_vit_h_4b8939.pth",
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                "md5": "4b8939a88964f0f4ff5f5b2642c598a6",
            },
        ),
    },
}
SAM_MODEL_CHOICES = tuple(spec["display"] for spec in SAM_MODEL_SPECS.values())
SAM_MODEL_ALIASES = {
    **{spec["display"]: key for key, spec in SAM_MODEL_SPECS.items()},
    "SAM ViT-B": "vit_b",
    "SAM ViT-L": "vit_l",
    "SAM ViT-H": "vit_h",
    "vit_b": "vit_b",
    "vit_l": "vit_l",
    "vit_h": "vit_h",
}
_DOWNLOAD_LOCK = threading.Lock()
_VERIFIED_MODELS = {}
_REMBG_SESSION_LOCK = threading.Lock()
_REMBG_SESSIONS = {}
_SAM_MODEL_LOCK = threading.Lock()
_SAM_MODEL_CACHE = {}


def _md5sum(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _migrate_legacy_erase_file(file_spec, path):
    """Move a verified pre-v1 erase model out of ``models/iopaint``.

    Earlier releases stored erase weights directly in the IOPaint model root.
    Only files named by ``MODEL_SPECS`` are considered, so unrelated user files
    are never touched.  Invalid legacy files are left in place and a clean copy
    is downloaded into the new ``erase`` directory.
    """

    if os.path.abspath(MODEL_DIRECTORY) != os.path.abspath(ERASE_MODEL_DIRECTORY):
        return

    legacy_path = os.path.join(MODEL_ROOT_DIRECTORY, file_spec["filename"])
    if legacy_path == path or not os.path.isfile(legacy_path) or os.path.exists(path):
        return

    try:
        if _md5sum(legacy_path) != file_spec["md5"]:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        os.replace(legacy_path, path)
        LOGGER.info("DvD IOPaint: migrated legacy erase model %s to %s", legacy_path, path)
    except OSError as error:
        LOGGER.warning("DvD IOPaint: could not migrate legacy model %s: %s", legacy_path, error)


def _ensure_model_file(model_name, file_spec, directory=None):
    directory = directory or MODEL_DIRECTORY
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, file_spec["filename"])

    if os.path.abspath(directory) == os.path.abspath(ERASE_MODEL_DIRECTORY):
        _migrate_legacy_erase_file(file_spec, path)

    with _DOWNLOAD_LOCK:
        if os.path.isfile(path):
            stat = os.stat(path)
            signature = (stat.st_size, stat.st_mtime_ns)
            if _VERIFIED_MODELS.get(path) == signature:
                return path
            if _md5sum(path) == file_spec["md5"]:
                _VERIFIED_MODELS[path] = signature
                return path

        if os.path.exists(path):
            os.remove(path)
            _VERIFIED_MODELS.pop(path, None)

        partial_path = path + ".download"
        if os.path.exists(partial_path):
            os.remove(partial_path)

        LOGGER.info("DvD IOPaint: downloading %s file to %s", model_name, path)
        try:
            download_url_to_file(file_spec["url"], partial_path, progress=True)
            downloaded_md5 = _md5sum(partial_path)
            if downloaded_md5 != file_spec["md5"]:
                raise RuntimeError(
                    f"Downloaded {model_name}/{file_spec['filename']} has MD5 "
                    f"{downloaded_md5}; expected {file_spec['md5']}."
                )
            os.replace(partial_path, path)
            stat = os.stat(path)
            _VERIFIED_MODELS[path] = (stat.st_size, stat.st_mtime_ns)
        except Exception as error:
            if os.path.exists(partial_path):
                os.remove(partial_path)
            raise RuntimeError(
                f"Could not download {model_name}/{file_spec['filename']} to "
                f"{path}. Check network access or place the verified file in "
                f"{directory}. Original error: {error}"
            ) from error

        LOGGER.info("DvD IOPaint: %s file download complete", model_name)
        return path


def _ensure_model_files(model_name):
    for file_spec in MODEL_SPECS[model_name]["files"]:
        _ensure_model_file(model_name, file_spec)


def _ensure_sam_model_file(model_name):
    """Ensure one SAM checkpoint exists in the interactive-segmentation dir."""

    for file_spec in SAM_MODEL_SPECS[model_name]["files"]:
        return _ensure_model_file(
            f"SAM {model_name}",
            file_spec,
            directory=INTERACTIVE_SEG_DIRECTORY,
        )
    raise RuntimeError(f"SAM model {model_name!r} has no checkpoint files")


@contextmanager
def _temporary_environment(name, value):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _resolve_model(model):
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    if model in MODEL_SPECS or model in OPENCV_CHOICES:
        return model
    raise ValueError(f"Unknown DvD IOPaint model: {model}")


def _resolve_mask_model(model):
    if model in MASK_MODEL_ALIASES:
        return MASK_MODEL_ALIASES[model]
    if model in MASK_MODEL_SPECS:
        return model
    raise ValueError(f"Unknown DvD IOPaint mask model: {model}")


def _resolve_sam_model(model):
    if model in SAM_MODEL_ALIASES:
        return SAM_MODEL_ALIASES[model]
    if model in SAM_MODEL_SPECS:
        return model
    raise ValueError(f"Unknown DvD IOPaint SAM model: {model}")


def _load_model_class(model_name):
    spec = MODEL_SPECS[model_name]
    module = importlib.import_module(
        f"dvd_iopaint_vendor.model.{spec['module']}"
    )
    return getattr(module, spec["class"])


def _get_rembg_session(model_name):
    """Create or reuse a rembg session in the selected mask directory."""

    spec = MASK_MODEL_SPECS[model_name]
    try:
        import rembg
    except ImportError as error:
        raise RuntimeError(
            "DvD IOPaint mask models require rembg. Install the optional "
            "mask dependencies from requirements-mask.txt, then restart ComfyUI."
        ) from error

    directory = os.path.abspath(spec["directory"])
    key = (spec["rembg_model"], directory)
    with _REMBG_SESSION_LOCK:
        session = _REMBG_SESSIONS.get(key)
        if session is not None:
            return session

        os.makedirs(directory, exist_ok=True)
        try:
            # rembg resolves its per-model files below U2NET_HOME. Scoping the
            # variable to construction keeps mask models separate from erase
            # models and does not change ComfyUI's global cache configuration.
            with _temporary_environment("U2NET_HOME", directory):
                session = rembg.new_session(model_name=spec["rembg_model"])
        except Exception as error:
            raise RuntimeError(
                f"Could not initialize DvD IOPaint mask model {model_name!r}. "
                f"The model is cached below {directory}. Original error: {error}"
            ) from error
        _REMBG_SESSIONS[key] = session
        return session


def _run_mask_model(model_name, image):
    """Return a grayscale uint8 foreground mask for one RGB image."""

    try:
        import rembg
    except ImportError as error:
        raise RuntimeError(
            "DvD IOPaint mask models require rembg. Install the optional "
            "mask dependencies from requirements-mask.txt, then restart ComfyUI."
        ) from error

    spec = MASK_MODEL_SPECS[model_name]
    session = _get_rembg_session(model_name)
    try:
        with _temporary_environment("U2NET_HOME", os.path.abspath(spec["directory"])):
            result = rembg.remove(
                image,
                session=session,
                only_mask=True,
                post_process_mask=False,
            )
    except Exception as error:
        raise RuntimeError(
            f"DvD IOPaint mask model {model_name!r} failed during inference: {error}"
        ) from error

    if isinstance(result, Image.Image):
        result = np.asarray(result.convert("L"))
    elif isinstance(result, bytes):
        result = np.asarray(Image.open(io.BytesIO(result)).convert("L"))
    else:
        result = np.asarray(result)
        if result.ndim == 3:
            if result.shape[-1] == 1:
                result = result[..., 0]
            else:
                result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    if result.ndim != 2:
        raise RuntimeError(
            f"DvD IOPaint mask model {model_name!r} returned an invalid mask shape: "
            f"{result.shape}"
        )
    return np.clip(result, 0, 255).astype(np.uint8)


def _postprocess_mask(mask, size, threshold, invert, feather):
    width, height = size
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {mask.shape}")
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    if mask.dtype != np.uint8:
        max_value = float(np.max(mask)) if mask.size else 0.0
        if max_value <= 1.0:
            mask = mask * 255.0
        mask = np.clip(mask, 0, 255).astype(np.uint8)

    if threshold > 0:
        mask = np.where(mask >= int(round(threshold * 255)), 255, 0).astype(np.uint8)
    if feather > 0:
        kernel = max(1, int(feather)) * 2 + 1
        mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    if invert:
        mask = 255 - mask
    return np.ascontiguousarray(mask)


def _parse_background_color(value):
    """Parse a user-selected RGB background color into float RGB values.

    The Mask Generator returns a three-channel ComfyUI IMAGE, so alpha values
    are intentionally not accepted here.  Hex (``#RRGGBB``/``#RGB``), common
    CSS color names, and comma-separated ``R,G,B`` values are supported.
    """

    text = "#000000" if value is None else str(value).strip()
    if not text:
        text = "#000000"

    try:
        if text.startswith("#"):
            digits = text[1:]
            if len(digits) == 3:
                digits = "".join(channel * 2 for channel in digits)
            if len(digits) != 6 or any(
                channel not in "0123456789abcdefABCDEF" for channel in digits
            ):
                raise ValueError
            rgb = tuple(
                int(digits[offset : offset + 2], 16) for offset in (0, 2, 4)
            )
        elif "," in text:
            channels = [int(channel.strip()) for channel in text.split(",")]
            if len(channels) != 3 or any(channel < 0 or channel > 255 for channel in channels):
                raise ValueError
            rgb = tuple(channels)
        else:
            rgb = ImageColor.getrgb(text)
            if len(rgb) != 3:
                raise ValueError
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid background_color {value!r}; use #RRGGBB, #RGB, a CSS "
            "color name, or R,G,B."
        ) from error

    return np.asarray(rgb, dtype=np.float32) / 255.0


def _tensor_image_to_uint8(image_tensor, index=0):
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(f"Expected an IMAGE tensor, got {type(image_tensor).__name__}")
    values = image_tensor.detach().cpu().float()
    if values.ndim != 4 or values.shape[-1] < 3:
        raise ValueError(
            f"Expected IMAGE tensor with shape [B,H,W,C], got {tuple(values.shape)}"
        )
    if index < 0 or index >= values.shape[0]:
        raise IndexError(f"Image batch index {index} is out of range")
    array = values[index].numpy()
    return np.clip(array[..., :3] * 255.0, 0, 255).astype(np.uint8)


def _tensor_mask_to_uint8(mask_tensor, size, index=0):
    if not isinstance(mask_tensor, torch.Tensor):
        raise TypeError(f"Expected a MASK tensor, got {type(mask_tensor).__name__}")
    values = mask_tensor.detach().cpu().float()
    if values.ndim == 4:
        # ComfyUI normally uses [B,H,W], but a few third-party nodes expose
        # masks as [B,1,H,W] or [B,H,W,1].  Accept both channel placements so
        # an external mask can be connected without a shape-specific failure.
        if values.shape[1] == 1:
            values = values[:, 0]
        elif values.shape[-1] == 1:
            values = values[..., 0]
        else:
            values = values.mean(dim=1)
    if values.ndim == 2:
        values = values.unsqueeze(0)
    if values.ndim != 3:
        raise ValueError(
            f"Expected MASK tensor with shape [B,H,W] or [B,1,H,W], got {tuple(values.shape)}"
        )
    if index < 0 or index >= values.shape[0]:
        raise IndexError(f"Mask batch index {index} is out of range")
    array = values[index].numpy()
    max_value = float(np.max(array)) if array.size else 0.0
    if max_value <= 1.0:
        array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    width, height = size
    if array.shape != (height, width):
        array = cv2.resize(array, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(array)


def _load_rgb_image(filename):
    path = folder_paths.get_annotated_filepath(filename)
    image = node_helpers.pillow(Image.open, path)
    image = node_helpers.pillow(ImageOps.exif_transpose, image)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_mask(filename, size):
    if not filename or not filename.strip():
        return np.zeros((size[1], size[0]), dtype=np.uint8)

    path = folder_paths.get_annotated_filepath(filename)
    mask = node_helpers.pillow(Image.open, path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask, dtype=np.uint8)


def _run_opencv(image, mask, model_name, radius):
    flag = cv2.INPAINT_TELEA if model_name == "opencv_telea" else cv2.INPAINT_NS
    return cv2.inpaint(image, mask, float(radius), flag)


def _blend_inpaint_edge(image, result, mask, edge_blur):
    """Blend a completed repair back into its source with a softened edge."""
    radius = max(0, int(edge_blur))
    if radius == 0:
        return result, mask

    kernel = radius * 2 + 1
    blend_mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    alpha = blend_mask.astype(np.float32)[..., None] / 255.0
    blended = result.astype(np.float32) * alpha + image.astype(np.float32) * (1.0 - alpha)
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8), blend_mask


def _move_runner_to_cpu(runner):
    for value in vars(runner).values():
        if isinstance(value, torch.nn.Module):
            value.to("cpu")


def _run_iopaint_model(model_name, image, mask):
    _ensure_model_files(model_name)
    device = model_management.get_torch_device()
    model_management.free_memory(2 * 1024 * 1024 * 1024, device)

    model_class = _load_model_class(model_name)
    schema = importlib.import_module("dvd_iopaint_vendor.schema")
    runner = None
    try:
        runner = model_class(device, no_half=False, fp16=True)
        result_bgr = runner(image, mask, schema.InpaintRequest())
        result_bgr = np.clip(result_bgr, 0, 255).astype(np.uint8)
        if result_bgr.ndim == 2:
            result = cv2.cvtColor(result_bgr, cv2.COLOR_GRAY2RGB)
        else:
            result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        if result.shape != image.shape:
            raise RuntimeError(
                f"{model_name} returned {result.shape}; expected {image.shape}."
            )
        result[mask == 0] = image[mask == 0]
        return result
    finally:
        if runner is not None:
            _move_runner_to_cpu(runner)
            del runner
        model_management.soft_empty_cache()


def _move_sam_model_to_cpu(model):
    try:
        model.to("cpu")
    except Exception as error:
        LOGGER.debug("DvD IOPaint: could not move cached SAM model to CPU: %s", error)


def _load_sam_predictor(model_name):
    """Load one vendored SAM1 predictor without changing ComfyUI packages."""

    model_name = _resolve_sam_model(model_name)
    device = model_management.get_torch_device()
    device_key = str(device)
    cache_key = (model_name, device_key)

    with _SAM_MODEL_LOCK:
        cached = _SAM_MODEL_CACHE.get(cache_key)
        if cached is None:
            # Keep only the selected checkpoint resident.  ViT-H is large and
            # retaining an earlier model here would unnecessarily evict user
            # workflows from VRAM when switching the combo box.
            for old_key, old_model in list(_SAM_MODEL_CACHE.items()):
                if old_key != cache_key:
                    _move_sam_model_to_cpu(old_model)
                    _SAM_MODEL_CACHE.pop(old_key, None)
            model_management.free_memory(512 * 1024 * 1024, device)
            checkpoint = _ensure_sam_model_file(model_name)
            try:
                sam_module = importlib.import_module(
                    "dvd_iopaint_vendor.segment_anything"
                )
                architecture = SAM_MODEL_SPECS[model_name]["architecture"]
                model = sam_module.sam_model_registry[architecture](
                    checkpoint=checkpoint
                )
                model.to(device)
                model.eval()
            except Exception as error:
                raise RuntimeError(
                    f"Could not load DvD IOPaint SAM model {model_name!r} "
                    f"from {checkpoint}: {error}"
                ) from error
            _SAM_MODEL_CACHE[cache_key] = model
            cached = model

        return importlib.import_module(
            "dvd_iopaint_vendor.segment_anything"
        ).SamPredictor(cached)


def _parse_sam_clicks(value):
    """Parse frontend click data in ``[[x, y, label], ...]`` form."""

    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                "SAM points must be JSON such as [[x, y, 1], [x, y, 0]]."
            ) from error
    if isinstance(value, dict):
        value = value.get("clicks", value.get("points", []))
    if not isinstance(value, (list, tuple)):
        raise ValueError("SAM points must be a list of [x, y, label] entries.")

    clicks = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(
                f"SAM point {index + 1} must contain x, y and label values."
            )
        try:
            x = float(item[0])
            y = float(item[1])
            label_float = float(item[2])
        except (TypeError, ValueError) as error:
            raise ValueError(f"SAM point {index + 1} contains a non-numeric value.") from error
        if not all(math.isfinite(number) for number in (x, y, label_float)):
            raise ValueError(f"SAM point {index + 1} contains a non-finite value.")
        label = int(label_float)
        if label not in (0, 1) or label_float != label:
            raise ValueError(f"SAM point {index + 1} label must be 0 or 1.")
        clicks.append([x, y, label])
    return clicks


def _postprocess_sam_mask(mask, size, blur=0, expand=0, invert=False):
    """Convert a SAM boolean mask to a ComfyUI-compatible uint8 mask."""

    width, height = size
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError(f"SAM returned an invalid mask shape: {values.shape}")
    if values.shape != (height, width):
        values = cv2.resize(
            values.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        )
    values = np.where(values > 0, 255, 0).astype(np.uint8)
    if expand:
        radius = max(0, int(expand))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        values = cv2.dilate(values, kernel)
    if blur:
        radius = max(1, int(blur))
        values = cv2.GaussianBlur(values, (radius * 2 + 1, radius * 2 + 1), 0)
    if invert:
        values = 255 - values
    return np.ascontiguousarray(values)


def _run_sam_segmentation(model_name, image, clicks, mask_blur, mask_expand, invert):
    predictor = _load_sam_predictor(model_name)
    points = np.asarray([[click[0], click[1]] for click in clicks], dtype=np.float32)
    # Keep prompts inside the actual source image.  This matters for a click
    # on a one-pixel border and for batched external images whose dimensions
    # may differ from the image used to draw the points in the frontend.
    if points.size:
        points[:, 0] = np.clip(points[:, 0], 0, max(0, image.shape[1] - 1))
        points[:, 1] = np.clip(points[:, 1], 0, max(0, image.shape[0] - 1))
    labels = np.asarray([click[2] for click in clicks], dtype=np.int32)
    predictor.set_image(image)
    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
    if masks is None or len(masks) == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    best_index = int(np.argmax(np.asarray(scores))) if scores is not None else 0
    return _postprocess_sam_mask(
        masks[best_index],
        (image.shape[1], image.shape[0]),
        blur=mask_blur,
        expand=mask_expand,
        invert=invert,
    )


class DvDIOPaintSAMInteractiveSegmentation:
    """Create a precise object mask from foreground/background clicks."""

    def __init__(self):
        self._preview = PreviewImage()

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            name
            for name in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, name))
        ]
        files = folder_paths.filter_files_content_types(files, ["image"])
        if not files:
            files = [""]
        return {
            "required": {
                "model": (
                    SAM_MODEL_CHOICES,
                    {
                        "default": SAM_MODEL_CHOICES[0],
                        "tooltip": "选择识别物体的 SAM 模型；模型越大通常越细致，但也更慢、更占显存。\nChoose the SAM model used to find the object. Larger models are usually more detailed, but slower and use more VRAM.",
                    },
                ),
                "image": (
                    sorted(files),
                    {
                        "image_upload": True,
                        "tooltip": "选择要识别的图片。也可以直接把图片拖到节点上。\nChoose the image to inspect. You can also drag an image onto the node.",
                    },
                ),
                "points": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": False,
                        "dynamicPrompts": False,
                        "tooltip": "记录你在图片上添加的选取点，由节点自动填写。\nStores the selection points placed on the image. The node fills this automatically.",
                    },
                ),
                "auto_run": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "开启后每加一个点都会立即识别；关闭后可先加多个点，再按播放按钮。\nWhen enabled, every new point runs SAM. Keep it off to place several points before pressing Play.",
                    },
                ),
                "mask_blur": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "display": "slider",
                        "tooltip": "让遮罩边缘更柔和。0 表示不模糊；数值越大，过渡范围越宽。\nSoftens the mask edge. 0 keeps it sharp; higher values make a wider transition.",
                    },
                ),
                "mask_expand": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "display": "slider",
                        "tooltip": "把遮罩边缘向外扩大指定像素，适合盖住物体残边。0 表示不扩大。\nGrows the mask outward by this many pixels to cover leftover object edges. 0 keeps the original size.",
                    },
                ),
                "invert": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "反转黑白区域：原本选中的部分变成未选中。\nSwaps selected and unselected areas of the mask.",
                    },
                ),
            },
            "optional": {
                "image_input": (
                    "IMAGE",
                    {
                        "tooltip": "接入其他节点的图片。接入后会优先使用这张图片。\nConnect an image from another node. A connected image takes priority over the file picker.",
                    },
                )
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "segment"
    CATEGORY = "DvD/Image"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "在图片上点击物体来生成遮罩：左键标记要选的区域，右键排除不要的区域。"
        "输出原图和遮罩，可连接到消除节点或其他节点。\n"
        "Click an object to create a mask: left-click areas to include and "
        "right-click areas to exclude. Outputs the source image and mask."
    )
    OUTPUT_TOOLTIPS = (
        "未经修改的原图。\nThe unchanged source image.",
        "根据点击位置生成的遮罩。\nThe mask created from your clicks.",
    )

    def segment(
        self,
        model,
        image,
        points,
        auto_run,
        image_input=None,
        mask_blur=0,
        mask_expand=0,
        invert=False,
        prompt=None,
        extra_pnginfo=None,
    ):
        del auto_run
        model_name = _resolve_sam_model(model)
        clicks = _parse_sam_clicks(points)
        if image_input is not None:
            if not isinstance(image_input, torch.Tensor) or image_input.ndim != 4:
                raise ValueError("image_input must be an IMAGE tensor with shape [B,H,W,C].")
            if image_input.shape[0] < 1:
                raise ValueError("image_input must contain at least one image.")
            image_arrays = [
                _tensor_image_to_uint8(image_input, index)
                for index in range(image_input.shape[0])
            ]
        else:
            if not image:
                raise ValueError("Choose an image or connect an IMAGE input before running DvD SAM.")
            image_arrays = [_load_rgb_image(image)]

        masks = []
        images = []
        for image_array in image_arrays:
            if clicks:
                mask = _run_sam_segmentation(
                    model_name,
                    image_array,
                    clicks,
                    mask_blur,
                    mask_expand,
                    invert,
                )
            else:
                mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
                if invert:
                    mask.fill(255)
            masks.append(mask.astype(np.float32) / 255.0)
            images.append(image_array.astype(np.float32) / 255.0)

        mask_tensor = torch.from_numpy(np.stack(masks, axis=0))
        image_tensor = torch.from_numpy(np.stack(images, axis=0))
        mask_preview_tensor = torch.from_numpy(
            np.repeat(mask_tensor.numpy()[..., None], 3, axis=-1)
        )
        source_preview = self._preview.save_images(
            image_tensor,
            filename_prefix="DvD_SAM_Source",
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
        mask_preview = self._preview.save_images(
            mask_preview_tensor,
            filename_prefix="DvD_SAM_Mask",
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
        return {
            "ui": {
                "dvd_sam_source": source_preview["ui"]["images"],
                "dvd_sam_mask": mask_preview["ui"]["images"],
            },
            "result": (image_tensor, mask_tensor),
        }

    @classmethod
    def IS_CHANGED(
        cls,
        model,
        image,
        points,
        auto_run,
        image_input=None,
        mask_blur=0,
        mask_expand=0,
        invert=False,
        prompt=None,
        extra_pnginfo=None,
    ):
        del auto_run, prompt, extra_pnginfo
        digest = hashlib.sha256()
        digest.update(
            f"{model}|{points}|{int(mask_blur)}|{int(mask_expand)}|{bool(invert)}".encode(
                "utf-8"
            )
        )
        if image and folder_paths.exists_annotated_filepath(image):
            path = folder_paths.get_annotated_filepath(image)
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        if isinstance(image_input, torch.Tensor):
            tensor = image_input.detach().cpu().contiguous()
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        image,
        model,
        points="[]",
        image_input=None,
        input_types=None,
        **kwargs,
    ):
        del kwargs
        if isinstance(model, str):
            try:
                _resolve_sam_model(model)
            except ValueError as error:
                return str(error)
        # A connected IMAGE takes precedence over the file widget.  ComfyUI
        # still sends the widget's stale filename during validation, so do not
        # reject a valid linked workflow merely because that old file was
        # removed from the input directory.
        linked_image = (
            isinstance(input_types, dict)
            and input_types.get("image_input") == "IMAGE"
        )
        if (
            image_input is None
            and not linked_image
            and image
            and not folder_paths.exists_annotated_filepath(image)
        ):
            return f"Invalid image file: {image}"
        try:
            _parse_sam_clicks(points)
        except ValueError as error:
            return str(error)
        return True


class DvDIOPaintMaskGenerator:
    """Generate a foreground/background mask with an IOPaint-style plugin."""

    def __init__(self):
        self._preview = PreviewImage()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "接入需要自动抠图的图片。\nConnect the image you want to cut out automatically.",
                    },
                ),
                "model": (
                    MASK_MODEL_CHOICES,
                    {
                        "default": MASK_MODEL_CHOICES[0],
                        "tooltip": "选择自动抠图模型。第一次使用某个模型时会自动下载。\nChoose the cutout model. Each model downloads automatically the first time it is used.",
                    },
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 2,
                        "display": "slider",
                        "tooltip": "控制哪些区域算作主体。调高通常会缩小选区，调低通常会保留更多边缘。\nControls what counts as the subject. Higher values usually shrink the selection; lower values usually keep more edge detail.",
                    },
                ),
                "invert": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "关闭时选择主体，开启后改为选择背景。\nOff selects the subject; on selects the background instead.",
                    },
                ),
                "feather": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "display": "slider",
                        "tooltip": "让抠图边缘更柔和。0 表示不处理；数值越大，边缘过渡越宽。\nSoftens the cutout edge. 0 leaves it unchanged; higher values make a wider transition.",
                    },
                ),
            },
            "optional": {
                "background_color": (
                    "STRING",
                    {
                        "default": "#000000",
                        "multiline": False,
                        "dynamicPrompts": False,
                        "tooltip": (
                            "设置抠图预览的背景颜色，可填写 #RRGGBB、颜色名称或 R,G,B。"
                            "这不会改变遮罩。\nSet the cutout preview background using #RRGGBB, "
                            "a color name, or R,G,B. This does not change the mask."
                        ),
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "foreground")
    FUNCTION = "generate"
    CATEGORY = "DvD/Image"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "自动找出图片主体并生成遮罩，同时输出带纯色背景的抠图预览。可反转为背景遮罩。\n"
        "Automatically finds the subject and creates a mask, plus a cutout preview "
        "on a solid background. The mask can be inverted to select the background."
    )
    OUTPUT_TOOLTIPS = (
        "自动生成的黑白遮罩，可连接到消除或合成节点。\nThe generated mask for erasing or compositing.",
        "带所选背景颜色的抠图预览，不包含透明通道。\nA cutout preview on the chosen background color; it has no alpha channel.",
    )

    def generate(
        self,
        image,
        model,
        threshold,
        invert,
        feather,
        prompt=None,
        extra_pnginfo=None,
        background_color="#000000",
    ):
        model_name = _resolve_mask_model(model)
        background_rgb = _parse_background_color(background_color)
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError(
                f"DvD IOPaint Mask Generator expects IMAGE [B,H,W,C], got {type(image).__name__}"
            )
        if image.shape[0] < 1:
            raise ValueError("DvD IOPaint Mask Generator received an empty image batch")

        masks = []
        foregrounds = []
        for index in range(image.shape[0]):
            rgb_image = _tensor_image_to_uint8(image, index)
            height, width = rgb_image.shape[:2]
            raw_mask = _run_mask_model(model_name, rgb_image)
            processed_mask = _postprocess_mask(
                raw_mask,
                (width, height),
                float(threshold),
                bool(invert),
                int(feather),
            )
            mask_float = processed_mask.astype(np.float32) / 255.0
            masks.append(mask_float)
            rgb_float = rgb_image.astype(np.float32) / 255.0
            foregrounds.append(
                rgb_float * mask_float[..., None]
                + background_rgb * (1.0 - mask_float[..., None])
            )

        mask_tensor = torch.from_numpy(np.stack(masks, axis=0))
        foreground_tensor = torch.from_numpy(np.stack(foregrounds, axis=0))
        preview = self._preview.save_images(
            foreground_tensor,
            filename_prefix="DvD_IOPaint_Mask",
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
        return {
            "ui": {"images": preview["ui"]["images"]},
            "result": (mask_tensor, foreground_tensor),
        }

    @classmethod
    def IS_CHANGED(
        cls,
        image,
        model,
        threshold,
        invert,
        feather,
        prompt=None,
        extra_pnginfo=None,
        background_color="#000000",
    ):
        del prompt, extra_pnginfo
        digest = hashlib.sha256()
        digest.update(
            f"{model}|{float(threshold):.6f}|{bool(invert)}|{int(feather)}|"
            f"{background_color}".encode("utf-8")
        )
        if isinstance(image, torch.Tensor):
            values = image.detach().cpu().contiguous()
            digest.update(str(tuple(values.shape)).encode("utf-8"))
            digest.update(values.numpy().tobytes())
        return digest.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, image, model, **kwargs):
        # During ComfyUI's prompt-validation pass, linked inputs are not
        # resolved to their runtime values yet.  ``get_input_data`` therefore
        # supplies a missing-value placeholder for a linked IMAGE (rather than
        # a torch.Tensor).  Type/shape validation belongs in ``generate`` once
        # the upstream node has executed; checking it here incorrectly rejects
        # every connected IMAGE before execution can start.
        del image, kwargs
        # Keep validating literal combo values, while allowing unresolved
        # linked/placeholder values to pass through ComfyUI's own link/type
        # validation.  This also preserves a useful error for stale workflows
        # that contain a model name no longer offered by this node.
        if isinstance(model, str):
            try:
                _resolve_mask_model(model)
            except ValueError as error:
                return str(error)
        return True


class DvDIOPaintInteractiveEraser:
    def __init__(self):
        self._preview = PreviewImage()

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            name
            for name in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, name))
        ]
        files = folder_paths.filter_files_content_types(files, ["image"])
        if not files:
            files = [""]

        return {
            "required": {
                "model": (
                    MODEL_CHOICES,
                    {
                        "default": DEFAULT_MODEL,
                        "tooltip": "选择用于补全被擦除区域的模型。第一次使用某个模型时会自动下载。\nChoose the model that fills the erased area. Each model downloads automatically on first use.",
                    },
                ),
                "image": (
                    sorted(files),
                    {
                        "image_upload": True,
                        "tooltip": "选择要处理的图片，也可以直接拖到节点上。外部图片接入后会优先使用外部图片。\nChoose an image or drag one onto the node. A connected image takes priority.",
                    },
                ),
                "mask": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "保存画布上的笔迹遮罩，由节点自动填写。\nStores the mask painted on the canvas. The node fills this automatically.",
                    },
                ),
                "brush_size": (
                    "INT",
                    {
                        "default": 48,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                        "display": "slider",
                        "tooltip": "画笔直径，单位是图片像素。鼠标在画布内时可按住 Alt 滚轮调整。\nBrush diameter in image pixels. Hold Alt and scroll while over the canvas to adjust it.",
                    },
                ),
                "auto_run": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "开启后松开画笔就自动消除；关闭后需要按播放按钮运行。\nWhen enabled, erasing starts after each stroke. When disabled, press Play to run.",
                    },
                ),
                "opencv_radius": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 32,
                        "step": 1,
                        "display": "slider",
                        "tooltip": "只对 OpenCV Telea 和 Navier-Stokes 有效。数值越大，参考周围颜色的范围越广；其他模型会忽略此项。\nOnly affects OpenCV Telea and Navier-Stokes. Higher values sample a wider nearby area; other models ignore this setting.",
                    },
                ),
                "edge_blur": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "display": "slider",
                        "tooltip": "让消除结果和原图的交界更柔和。0 不处理；大于 0 时也会作用于接入的外部遮罩。\nSoftens the join between the repaired area and the source. 0 leaves it unchanged; values above 0 also affect a connected mask.",
                    },
                ),
            },
            "optional": {
                "image_input": (
                    "IMAGE",
                    {
                        "tooltip": "接入其他节点的图片。接入后会优先使用它，而不是文件选项中的图片。\nConnect an image from another node. It takes priority over the selected file.",
                    },
                ),
                "mask_input": (
                    "MASK",
                    {
                        "tooltip": "接入其他节点的遮罩。它会与画布上的笔迹合并。\nConnect a mask from another node. It is combined with strokes painted here.",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "erase"
    CATEGORY = "DvD/Image"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "直接在图片上涂抹要消除的内容，节点会自动补全该区域。也可以接入其他节点的图片和遮罩。\n"
        "Paint over anything you want to remove and the node fills that area. "
        "Images and masks from other nodes can also be connected."
    )
    OUTPUT_TOOLTIPS = (
        "完成消除后的图片。\nThe repaired image.",
        "本次实际使用的遮罩；开启边缘模糊时会输出柔化后的遮罩。\nThe mask used for this repair; when edge blur is enabled, this is the softened mask.",
    )

    def erase(
        self,
        model,
        image,
        mask,
        brush_size,
        auto_run,
        opencv_radius,
        edge_blur=0,
        prompt=None,
        extra_pnginfo=None,
        image_input=None,
        mask_input=None,
    ):
        del brush_size, auto_run
        if image_input is None and not image:
            raise ValueError("Choose an image before running DvD IOPaint.")

        model_name = _resolve_model(model)
        if image_input is not None:
            if not isinstance(image_input, torch.Tensor) or image_input.ndim != 4:
                raise ValueError("image_input must be an IMAGE tensor with shape [B,H,W,C].")
            if image_input.shape[0] < 1:
                raise ValueError("image_input must contain at least one image.")
            image_arrays = [
                _tensor_image_to_uint8(image_input, index)
                for index in range(image_input.shape[0])
            ]
        else:
            image_arrays = [_load_rgb_image(image)]

        external_mask_batch = 0
        if isinstance(mask_input, torch.Tensor):
            values = mask_input.detach()
            if values.ndim == 4 or values.ndim == 3:
                external_mask_batch = int(values.shape[0])
            elif values.ndim == 2:
                external_mask_batch = 1

        result_arrays = []
        mask_arrays = []
        for index, image_array in enumerate(image_arrays):
            height, width = image_array.shape[:2]
            mask_array = _load_mask(mask, (width, height))
            if mask_input is not None:
                mask_index = index if external_mask_batch > index else 0
                external_mask = _tensor_mask_to_uint8(
                    mask_input, (width, height), mask_index
                )
                mask_array = np.maximum(mask_array, external_mask)

            if not np.any(mask_array):
                result_array = image_array.copy()
            elif model_name in MODEL_SPECS:
                result_array = _run_iopaint_model(
                    model_name, image_array, mask_array
                )
            else:
                result_array = _run_opencv(
                    image_array, mask_array, model_name, opencv_radius
                )
            result_array, output_mask = _blend_inpaint_edge(
                image_array, result_array, mask_array, edge_blur
            )
            result_arrays.append(result_array.astype(np.float32) / 255.0)
            mask_arrays.append(output_mask.astype(np.float32) / 255.0)

        result_tensor = torch.from_numpy(np.stack(result_arrays, axis=0))
        mask_tensor = torch.from_numpy(np.stack(mask_arrays, axis=0))
        # Keep an explicit copy of the actual source used for this execution.
        # When ``image_input`` is connected it can differ from the node's file
        # widget, so the frontend must not use its stale canvas as the
        # "before" side of the comparison.
        source_tensor = torch.from_numpy(
            np.stack([array.astype(np.float32) / 255.0 for array in image_arrays], axis=0)
        )
        preview = self._preview.save_images(
            result_tensor,
            filename_prefix="DvD_IOPaint",
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
        source_preview = None
        if image_input is not None:
            source_preview = self._preview.save_images(
                source_tensor,
                filename_prefix="DvD_IOPaint_Before",
                prompt=prompt,
                extra_pnginfo=extra_pnginfo,
            )
        ui = {"dvd_iopaint_result": preview["ui"]["images"]}
        if source_preview is not None:
            ui["dvd_iopaint_before"] = source_preview["ui"]["images"]
        return {
            "ui": ui,
            "result": (result_tensor, mask_tensor),
        }

    @classmethod
    def IS_CHANGED(
        cls,
        model,
        image,
        mask,
        brush_size,
        auto_run,
        opencv_radius,
        edge_blur=0,
        prompt=None,
        extra_pnginfo=None,
        image_input=None,
        mask_input=None,
    ):
        del brush_size, auto_run, prompt, extra_pnginfo
        digest = hashlib.sha256()
        digest.update(f"{model}|{opencv_radius}|{int(edge_blur)}".encode("utf-8"))
        for filename in (image, mask):
            if filename and folder_paths.exists_annotated_filepath(filename):
                path = folder_paths.get_annotated_filepath(filename)
                with open(path, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
        for value in (image_input, mask_input):
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tuple(tensor.shape)).encode("utf-8"))
                digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        image,
        mask,
        image_input=None,
        mask_input=None,
        input_types=None,
        **kwargs,
    ):
        del mask_input, kwargs
        linked_image = (
            isinstance(input_types, dict)
            and input_types.get("image_input") == "IMAGE"
        )
        if image_input is None and not linked_image and (
            not image or not folder_paths.exists_annotated_filepath(image)
        ):
            return f"Invalid image file: {image}"
        if mask and not folder_paths.exists_annotated_filepath(mask):
            return f"Invalid mask file: {mask}"
        return True


NODE_CLASS_MAPPINGS = {
    "DvD_IOPaint_SAM_Interactive_Segmentation": DvDIOPaintSAMInteractiveSegmentation,
    "DvD_IOPaint_Mask_Generator": DvDIOPaintMaskGenerator,
    "DvD_IOPaint_Interactive_Eraser": DvDIOPaintInteractiveEraser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DvD_IOPaint_SAM_Interactive_Segmentation": "DvD IOPaint SAM Interactive Segmentation",
    "DvD_IOPaint_Mask_Generator": "DvD IOPaint Mask Generator",
    "DvD_IOPaint_Interactive_Eraser": "DvD IOPaint Interactive Eraser",
}
