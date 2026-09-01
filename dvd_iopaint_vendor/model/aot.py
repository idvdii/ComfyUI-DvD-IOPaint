import os

import cv2
import numpy as np
import torch

from dvd_iopaint_vendor.helper import (
    download_model,
    get_cache_path_by_url,
    load_jit_model,
)
from dvd_iopaint_vendor.schema import InpaintRequest
from .base import InpaintModel


AOT_MODEL_URL = os.environ.get(
    "AOT_MODEL_URL",
    "https://huggingface.co/ogkalu/aot-inpainting/resolve/"
    "42ffc84ff1bd46dd95f1c5a41e83ee7e98f39189/aot_traced.pt",
)
AOT_MODEL_MD5 = os.environ.get(
    "AOT_MODEL_MD5", "82350e9ce753fbdf59bc9250bcf4dd73"
)


class AOT(InpaintModel):
    name = "aot"
    pad_mod = 4
    is_erase_model = True

    def init_model(self, device, **kwargs):
        self.model = load_jit_model(AOT_MODEL_URL, device, AOT_MODEL_MD5)

    @staticmethod
    def download():
        download_model(AOT_MODEL_URL, AOT_MODEL_MD5)

    @staticmethod
    def is_downloaded() -> bool:
        return os.path.exists(get_cache_path_by_url(AOT_MODEL_URL))

    def forward(self, image, mask, config: InpaintRequest):
        del config
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image_bgr)
        ).permute(2, 0, 1).unsqueeze(0).float()
        image_tensor = image_tensor.to(self.device) / 127.5 - 1.0

        # InpaintModel pads 2-D masks to [H, W, 1]; normalize both that
        # representation and direct 2-D calls to the [N, 1, H, W] layout
        # expected by the traced AOT network.
        mask_2d = mask[:, :, 0] if mask.ndim == 3 else mask
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_2d))
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        mask_tensor = (mask_tensor > 127).to(image_tensor.dtype)
        image_tensor *= 1 - mask_tensor

        result = self.model(image_tensor, mask_tensor)
        result = result[0].permute(1, 2, 0).float().cpu().numpy()
        return np.clip((result + 1.0) * 127.5, 0, 255).astype(np.uint8)
