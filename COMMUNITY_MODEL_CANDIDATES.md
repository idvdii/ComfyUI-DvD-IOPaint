# Community model candidates

Research list only. None of these models are exposed by the DvD node yet.
Licenses, exact revisions, download sizes, memory requirements, and output
quality must be checked again before integration.

## Direct inpainting checkpoints

| Candidate | Focus | Source status | Expected integration |
| --- | --- | --- | --- |
| `runwayml/stable-diffusion-inpainting` | General SD 1.5 inpainting | Listed by IOPaint documentation | Diffusers pipeline, roughly 4-5 GiB |
| `Uminosachi/realisticVisionV51_v51VAE-inpainting` | Photorealistic SD 1.5 | Listed by current IOPaint source | Diffusers pipeline, roughly 4-5 GiB |
| `redstonehero/dreamshaper-inpainting` | Illustration and general creative images | Listed by current IOPaint source | Diffusers pipeline, roughly 4-5 GiB |
| `Sanster/anything-4.0-inpainting` | Anime SD 1.5 | Listed by IOPaint documentation and source | Diffusers pipeline, roughly 4-5 GiB |
| `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | General SDXL inpainting | Listed by IOPaint documentation and source | Diffusers SDXL pipeline, roughly 10-14 GiB |
| `diffusionbee/fooocus_inpainting` | Fooocus-style patching | Listed by IOPaint documentation | Requires a compatibility test before estimating footprint |

## BrushNet and task-guided candidates

| Candidate | Focus | Source status | Expected integration |
| --- | --- | --- | --- |
| `Sanster/PowerPaint-V1-stable-diffusion-inpainting` | Object removal, context fill, shape-guided fill and outpainting | Supported by current IOPaint source | Adds prompts and task selection; substantially heavier than erase models |
| `Sanster/brushnet_random_mask` | General BrushNet inpainting | Listed by current IOPaint source | Requires a compatible SD 1.5 base model |
| `Sanster/brushnet_segmentation_mask` | Segmentation-shaped masks | Listed by current IOPaint source | Requires a compatible SD 1.5 base model |
| `Regulus0725/random_mask_brushnet_ckpt_sdxl_regulus_v1` | SDXL BrushNet inpainting | Listed by current IOPaint source | Requires an SDXL base model and high VRAM |

## Base models usable through BrushNet or PowerPaint

| Candidate | Focus | Source status | Note |
| --- | --- | --- | --- |
| `RunDiffusion/Juggernaut-XI-v11` | Photorealistic SDXL | Listed by current IOPaint source | Not a standalone erase model |
| `SG161222/RealVisXL_V5.0` | Photorealistic SDXL | Listed by current IOPaint source | Not a standalone erase model |
| `eienmojiki/Anything-XL` | Anime SDXL | Listed by current IOPaint source | Not a standalone erase model |

## Suggested screening order

1. Test the four direct SD 1.5 inpainting candidates first. They have the
   smallest integration change and cover photo, illustration, and anime use.
2. Compare PowerPaint with the direct models only if prompt or task controls
   are acceptable in the single-node UI.
3. Consider SDXL and SDXL BrushNet last because their downloads and VRAM use
   are much larger and conflict with the current instant, stroke-by-stroke
   interaction goal.
