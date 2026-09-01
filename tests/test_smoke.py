import importlib.util
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

COMFYUI_DIRECTORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFYUI_DIRECTORY))
import folder_paths


PLUGIN_DIRECTORY = Path(__file__).resolve().parents[1]


def load_plugin():
    spec = importlib.util.spec_from_file_location(
        "dvd_iopaint_test",
        PLUGIN_DIRECTORY / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIRECTORY)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    with tempfile.TemporaryDirectory() as directory:
        test_directory = Path(directory)
        input_directory = test_directory / "input"
        temp_directory = test_directory / "temp"
        input_directory.mkdir()
        temp_directory.mkdir()
        folder_paths.set_input_directory(str(input_directory))
        folder_paths.set_temp_directory(str(temp_directory))

        image = np.zeros((96, 128, 3), dtype=np.uint8)
        image[:, :, 0] = np.arange(128, dtype=np.uint8)[None, :]
        image[32:64, 48:80] = (255, 255, 255)
        mask = np.zeros((96, 128), dtype=np.uint8)
        mask[32:64, 48:80] = 255
        Image.fromarray(image).save(input_directory / "source.png")
        Image.fromarray(mask).save(input_directory / "mask.png")

        plugin = load_plugin()
        backend = sys.modules[f"{plugin.__name__}.nodes"]
        assert len(backend.MODEL_CHOICES) == 11
        assert backend._resolve_model("LaMa") == "lama"
        assert backend._resolve_model("LaMa (~196 MiB)") == "lama"
        assert backend._resolve_model("AOT Manga/Anime (~22 MiB)") == "aot"
        for model_name in backend.MODEL_SPECS:
            assert backend._load_model_class(model_name) is not None

        base = importlib.import_module("dvd_iopaint_vendor.model.base")
        schema = importlib.import_module("dvd_iopaint_vendor.schema")

        class ReadOnlyCropModel(base.InpaintModel):
            name = "read_only_crop_test"

            def init_model(self, device, **kwargs):
                del device, kwargs

            @staticmethod
            def is_downloaded():
                return True

            @staticmethod
            def download():
                return None

            def forward(self, image, mask, config):
                del mask, config
                return image[:, :, ::-1].copy()

            def _run_box(self, image, mask, box, config):
                del image, mask, box, config
                crop = np.full((2, 2, 3), 127, dtype=np.uint8)
                return crop, [1, 1, 3, 3]

        readonly_image = np.asarray(
            Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        )
        assert not readonly_image.flags.writeable
        crop_mask = np.zeros((10, 10), dtype=np.uint8)
        crop_mask[1:3, 1:3] = 255
        crop_result = ReadOnlyCropModel("cpu")(
            readonly_image,
            crop_mask,
            schema.InpaintRequest(hd_strategy_crop_trigger_size=4),
        )
        assert crop_result.flags.writeable
        assert np.all(crop_result[1:3, 1:3] == 127)

        model_directory = test_directory / "models"
        fixture = b"dvd-iopaint-download-fixture"
        fixture_spec = {
            "filename": "fixture.bin",
            "url": "https://example.invalid/fixture.bin",
            "md5": hashlib.md5(fixture).hexdigest(),
        }
        original_model_directory = backend.MODEL_DIRECTORY
        original_download = backend.download_url_to_file
        download_count = 0

        def mock_download(url, path, progress):
            nonlocal download_count
            del url, progress
            download_count += 1
            Path(path).write_bytes(fixture)

        backend.MODEL_DIRECTORY = str(model_directory)
        backend.download_url_to_file = mock_download
        try:
            downloaded_path = Path(
                backend._ensure_model_file("fixture", fixture_spec)
            )
            assert downloaded_path.read_bytes() == fixture
            backend._ensure_model_file("fixture", fixture_spec)
            assert download_count == 1
            downloaded_path.write_bytes(b"corrupted")
            backend._ensure_model_file("fixture", fixture_spec)
            assert downloaded_path.read_bytes() == fixture
            assert download_count == 2
        finally:
            backend.MODEL_DIRECTORY = original_model_directory
            backend.download_url_to_file = original_download
            backend._VERIFIED_MODELS.clear()

        node_class = plugin.NODE_CLASS_MAPPINGS["DvD_IOPaint_Interactive_Eraser"]
        node = node_class()
        output = node.erase(
            "OpenCV Telea",
            "source.png",
            "mask.png",
            32,
            False,
            4,
        )

        result, result_mask = output["result"]
        assert tuple(result.shape) == (1, 96, 128, 3)
        assert tuple(result_mask.shape) == (1, 96, 128)
        assert float(result_mask.max()) == 1.0
        assert "images" not in output["ui"]
        preview = output["ui"]["dvd_iopaint_result"][0]
        preview_path = temp_directory / preview["subfolder"] / preview["filename"]
        assert preview_path.is_file()
        assert node_class.VALIDATE_INPUTS("source.png", "mask.png") is True
        changed = node_class.IS_CHANGED(
            "OpenCV Telea",
            "source.png",
            "mask.png",
            32,
            False,
            4,
            prompt={},
            extra_pnginfo={},
        )
        assert len(changed) == 64
        print("DvD IOPaint OpenCV smoke test passed")


if __name__ == "__main__":
    main()
