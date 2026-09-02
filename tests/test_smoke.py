import importlib.util
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
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
        assert Path(backend.MODEL_DIRECTORY).name == "erase"
        assert Path(backend.MASK_MODEL_DIRECTORY).name == "mask"
        assert Path(backend.MASK_REMBG_DIRECTORY).name == "removebg"
        assert Path(backend.MASK_ANIME_DIRECTORY).name == "anime_seg"
        assert Path(backend.INTERACTIVE_SEG_DIRECTORY).name == "interactive_seg"
        assert len(backend.SAM_MODEL_CHOICES) == 3
        sam_inputs = plugin.NODE_CLASS_MAPPINGS[
            "DvD_IOPaint_SAM_Interactive_Segmentation"
        ].INPUT_TYPES()["required"]
        assert sam_inputs["auto_run"][1]["default"] is False
        assert "开启后每加一个点" in sam_inputs["auto_run"][1]["tooltip"]
        assert "\n" in sam_inputs["auto_run"][1]["tooltip"]
        assert sam_inputs["mask_expand"][1]["min"] == 0
        assert sam_inputs["mask_expand"][1]["max"] == 256
        eraser_class = plugin.NODE_CLASS_MAPPINGS["DvD_IOPaint_Interactive_Eraser"]
        eraser_inputs = eraser_class.INPUT_TYPES()
        assert eraser_inputs["required"]["edge_blur"][1]["default"] == 0
        assert eraser_inputs["required"]["edge_blur"][1]["max"] == 64
        assert "tooltip" in eraser_inputs["optional"]["mask_input"][1]
        assert len(eraser_class.OUTPUT_TOOLTIPS) == len(eraser_class.RETURN_TYPES)
        assert "\n" in eraser_class.DESCRIPTION
        assert backend._resolve_model("LaMa") == "lama"
        assert backend._resolve_model("LaMa (~196 MiB)") == "lama"
        assert backend._resolve_model("AOT Manga/Anime (~22 MiB)") == "aot"
        assert backend._resolve_mask_model("Anime Segmentation") == "anime_seg"
        assert backend._resolve_mask_model("U2NetP") == "removebg_u2netp"
        assert backend._resolve_sam_model("SAM ViT-B (~375 MiB)") == "vit_b"
        assert backend._parse_sam_clicks("[[12, 23, 1], [4, 5, 0]]") == [
            [12.0, 23.0, 1],
            [4.0, 5.0, 0],
        ]
        seed_mask = np.zeros((11, 11), dtype=np.uint8)
        seed_mask[5, 5] = 255
        expanded_mask = backend._postprocess_sam_mask(
            seed_mask, (11, 11), expand=2
        )
        assert expanded_mask[5, 3] == 255
        assert expanded_mask[5, 2] == 0
        source = np.zeros((9, 9, 3), dtype=np.uint8)
        repaired = np.full_like(source, 255)
        hard_mask = np.zeros((9, 9), dtype=np.uint8)
        hard_mask[3:6, 3:6] = 255
        blended, soft_mask = backend._blend_inpaint_edge(
            source, repaired, hard_mask, 1
        )
        assert soft_mask[4, 4] == 255
        assert 0 < soft_mask[2, 4] < 255
        assert np.array_equal(blended[..., 0], soft_mask)
        unchanged, unchanged_mask = backend._blend_inpaint_edge(
            source, repaired, hard_mask, 0
        )
        assert unchanged is repaired and unchanged_mask is hard_mask
        assert np.allclose(
            backend._parse_background_color("#0f0"),
            np.asarray([0.0, 1.0, 0.0]),
        )
        assert np.allclose(
            backend._parse_background_color("255, 128, 0"),
            np.asarray([1.0, 128.0 / 255.0, 0.0]),
        )
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

        original_root_directory = backend.MODEL_ROOT_DIRECTORY
        original_erase_directory = backend.ERASE_MODEL_DIRECTORY
        migration_root = test_directory / "legacy_iopaint"
        migration_erase = migration_root / "erase"
        migration_root.mkdir()
        legacy_path = migration_root / "fixture.bin"
        legacy_path.write_bytes(fixture)
        backend.MODEL_ROOT_DIRECTORY = str(migration_root)
        backend.ERASE_MODEL_DIRECTORY = str(migration_erase)
        backend.MODEL_DIRECTORY = str(migration_erase)
        try:
            migrated_path = Path(backend._ensure_model_file("fixture", fixture_spec))
            assert migrated_path == migration_erase / "fixture.bin"
            assert migrated_path.read_bytes() == fixture
            assert not legacy_path.exists()
        finally:
            backend.MODEL_ROOT_DIRECTORY = original_root_directory
            backend.ERASE_MODEL_DIRECTORY = original_erase_directory
            backend.MODEL_DIRECTORY = original_model_directory
            backend._VERIFIED_MODELS.clear()

        # Exercise the mask node without downloading a real ONNX checkpoint.
        # The production path still uses rembg's first-use download and cache;
        # this fixture only verifies tensor conversion and node contracts.
        import rembg

        original_get_session = backend._get_rembg_session
        original_remove = rembg.remove

        def fake_get_session(model_name):
            assert model_name == "anime_seg"
            return object()

        def fake_remove(data, **kwargs):
            assert kwargs["only_mask"] is True
            height, width = data.shape[:2]
            result = np.zeros((height, width), dtype=np.uint8)
            result[height // 4 : height * 3 // 4, width // 4 : width * 3 // 4] = 255
            return Image.fromarray(result, mode="L")

        backend._get_rembg_session = fake_get_session
        rembg.remove = fake_remove
        try:
            mask_node = plugin.NODE_CLASS_MAPPINGS["DvD_IOPaint_Mask_Generator"]()
            mask_output = mask_node.generate(
                torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0),
                "Anime Segmentation / ISNet Anime (~170 MiB)",
                0.5,
                False,
                0,
                background_color="#ff0000",
                prompt={},
                extra_pnginfo={},
            )
            generated_mask, foreground = mask_output["result"]
            assert tuple(generated_mask.shape) == (1, 96, 128)
            assert tuple(foreground.shape) == (1, 96, 128, 3)
            assert float(generated_mask.max()) == 1.0
            assert torch.allclose(
                foreground[0, 0, 0],
                torch.tensor([1.0, 0.0, 0.0]),
            )
            assert mask_output["ui"]["images"]
        finally:
            backend._get_rembg_session = original_get_session
            rembg.remove = original_remove

        # Exercise the interactive SAM node contract with a lightweight fake
        # predictor so CI never downloads a real checkpoint.
        original_run_sam = backend._run_sam_segmentation

        def fake_run_sam(model_name, rgb_image, clicks, blur, offset, invert):
            assert model_name == "vit_b"
            assert clicks == [[64.0, 48.0, 1]]
            assert blur == 0 and offset == 0 and invert is False
            result = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
            result[24:72, 32:96] = 255
            return result

        backend._run_sam_segmentation = fake_run_sam
        try:
            sam_node_class = plugin.NODE_CLASS_MAPPINGS[
                "DvD_IOPaint_SAM_Interactive_Segmentation"
            ]
            sam_node = sam_node_class()
            sam_output = sam_node.segment(
                "SAM ViT-B (~375 MiB)",
                "source.png",
                "[[64,48,1]]",
                False,
                prompt={},
                extra_pnginfo={},
            )
            sam_image, sam_mask = sam_output["result"]
            assert tuple(sam_mask.shape) == (1, 96, 128)
            assert tuple(sam_image.shape) == (1, 96, 128, 3)
            assert float(sam_mask.max()) == 1.0
            assert sam_output["ui"]["dvd_sam_source"]
            assert sam_output["ui"]["dvd_sam_mask"]
            assert sam_node_class.VALIDATE_INPUTS(
                "source.png", "SAM ViT-B (~375 MiB)", "[[64,48,1]]"
            ) is True
            assert sam_node_class.VALIDATE_INPUTS(
                "stale-file.png",
                "SAM ViT-B (~375 MiB)",
                "[[64,48,1]]",
                image_input=torch.zeros((1, 8, 8, 3)),
            ) is True
            assert sam_node_class.VALIDATE_INPUTS(
                "stale-file.png",
                "SAM ViT-B (~375 MiB)",
                "[[64,48,1]]",
                input_types={"image_input": "IMAGE"},
            ) is True
        finally:
            backend._run_sam_segmentation = original_run_sam

        # Linked IMAGE values are unresolved during prompt validation.  They
        # must not be mistaken for a missing runtime tensor.
        mask_node_class = plugin.NODE_CLASS_MAPPINGS["DvD_IOPaint_Mask_Generator"]
        linked_image = ["12", 0]
        assert (
            mask_node_class.VALIDATE_INPUTS(
                linked_image,
                "Anime Segmentation / ISNet Anime (~170 MiB)",
            )
            is True
        )
        # ``get_input_data`` currently presents that unresolved link as None
        # to VALIDATE_INPUTS; this representation must be accepted as well.
        assert (
            mask_node_class.VALIDATE_INPUTS(
                None,
                "Anime Segmentation / ISNet Anime (~170 MiB)",
            )
            is True
        )

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
        assert "dvd_iopaint_before" not in output["ui"]
        assert node_class.VALIDATE_INPUTS("source.png", "mask.png") is True
        input_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        external_mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)
        external_output = node.erase(
            "OpenCV Telea",
            "",
            "",
            32,
            False,
            4,
            image_input=input_tensor,
            mask_input=external_mask,
        )
        external_result, external_result_mask = external_output["result"]
        assert tuple(external_result.shape) == (1, 96, 128, 3)
        assert tuple(external_result_mask.shape) == (1, 96, 128)
        assert external_output["ui"]["dvd_iopaint_before"]
        external_before = external_output["ui"]["dvd_iopaint_before"][0]
        external_before_path = (
            temp_directory
            / external_before["subfolder"]
            / external_before["filename"]
        )
        external_before_pixels = np.asarray(Image.open(external_before_path).convert("RGB"))
        assert tuple(external_before_pixels[0, 0]) == tuple(image[0, 0])
        distinct_external = np.full_like(image, (17, 33, 55))
        distinct_external_output = node.erase(
            "OpenCV Telea",
            "",
            "",
            32,
            False,
            4,
            image_input=torch.from_numpy(distinct_external.astype(np.float32) / 255.0).unsqueeze(0),
        )
        distinct_before = distinct_external_output["ui"]["dvd_iopaint_before"][0]
        distinct_before_path = (
            temp_directory
            / distinct_before["subfolder"]
            / distinct_before["filename"]
        )
        distinct_before_pixels = np.asarray(Image.open(distinct_before_path).convert("RGB"))
        assert tuple(distinct_before_pixels[0, 0]) == (17, 33, 55)
        assert node_class.VALIDATE_INPUTS("", "", image_input=input_tensor) is True
        assert node_class.VALIDATE_INPUTS(
            "stale-file.png",
            "",
            input_types={"image_input": "IMAGE"},
        ) is True
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
