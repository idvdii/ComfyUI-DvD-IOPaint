import hashlib
import importlib
import logging
import os
import sys
import threading

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.hub import download_url_to_file

import comfy.model_management as model_management
import folder_paths
import node_helpers
from nodes import PreviewImage


LOGGER = logging.getLogger(__name__)
MODEL_DIRECTORY = os.path.join(folder_paths.models_dir, "iopaint")
PLUGIN_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
if PLUGIN_DIRECTORY not in sys.path:
    sys.path.insert(0, PLUGIN_DIRECTORY)
os.environ["DVD_IOPAINT_MODEL_DIR"] = MODEL_DIRECTORY

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
_DOWNLOAD_LOCK = threading.Lock()
_VERIFIED_MODELS = {}


def _md5sum(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_model_file(model_name, file_spec):
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)
    path = os.path.join(MODEL_DIRECTORY, file_spec["filename"])

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
                f"{MODEL_DIRECTORY}. Original error: {error}"
            ) from error

        LOGGER.info("DvD IOPaint: %s file download complete", model_name)
        return path


def _ensure_model_files(model_name):
    for file_spec in MODEL_SPECS[model_name]["files"]:
        _ensure_model_file(model_name, file_spec)


def _resolve_model(model):
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    if model in MODEL_SPECS or model in OPENCV_CHOICES:
        return model
    raise ValueError(f"Unknown DvD IOPaint model: {model}")


def _load_model_class(model_name):
    spec = MODEL_SPECS[model_name]
    module = importlib.import_module(
        f"dvd_iopaint_vendor.model.{spec['module']}"
    )
    return getattr(module, spec["class"])


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
                "model": (MODEL_CHOICES, {"default": DEFAULT_MODEL}),
                "image": (sorted(files), {"image_upload": True}),
                "mask": ("STRING", {"default": ""}),
                "brush_size": (
                    "INT",
                    {"default": 48, "min": 1, "max": 512, "step": 1, "display": "slider"},
                ),
                "auto_run": ("BOOLEAN", {"default": True}),
                "opencv_radius": (
                    "INT",
                    {"default": 4, "min": 1, "max": 32, "step": 1, "display": "slider"},
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "erase"
    CATEGORY = "DvD/Image"
    OUTPUT_NODE = True
    DESCRIPTION = "Paint a mask directly on the node and run IOPaint object removal."

    def erase(
        self,
        model,
        image,
        mask,
        brush_size,
        auto_run,
        opencv_radius,
        prompt=None,
        extra_pnginfo=None,
    ):
        del brush_size, auto_run
        if not image:
            raise ValueError("Choose an image before running DvD IOPaint.")

        image_array = _load_rgb_image(image)
        height, width = image_array.shape[:2]
        mask_array = _load_mask(mask, (width, height))
        model_name = _resolve_model(model)

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

        result_tensor = torch.from_numpy(
            result_array.astype(np.float32) / 255.0
        ).unsqueeze(0)
        mask_tensor = torch.from_numpy(
            mask_array.astype(np.float32) / 255.0
        ).unsqueeze(0)
        preview = self._preview.save_images(
            result_tensor,
            filename_prefix="DvD_IOPaint",
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
        return {
            "ui": {"dvd_iopaint_result": preview["ui"]["images"]},
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
        prompt=None,
        extra_pnginfo=None,
    ):
        del brush_size, auto_run, prompt, extra_pnginfo
        digest = hashlib.sha256()
        digest.update(f"{model}|{opencv_radius}".encode("utf-8"))
        for filename in (image, mask):
            if filename and folder_paths.exists_annotated_filepath(filename):
                path = folder_paths.get_annotated_filepath(filename)
                with open(path, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, image, mask, **kwargs):
        if not image or not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        if mask and not folder_paths.exists_annotated_filepath(mask):
            return f"Invalid mask file: {mask}"
        return True


NODE_CLASS_MAPPINGS = {
    "DvD_IOPaint_Interactive_Eraser": DvDIOPaintInteractiveEraser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DvD_IOPaint_Interactive_Eraser": "DvD IOPaint Interactive Eraser",
}
