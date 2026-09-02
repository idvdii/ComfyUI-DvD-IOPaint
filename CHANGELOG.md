# Changelog / 更新日志

## 2.0.0 - 2026-09-02

### Added / 新增

- Added **DvD IOPaint Mask Generator** with anime segmentation and RemoveBG
  models. It outputs a reusable mask and a solid-background cutout preview.
  / 新增自动抠图节点，支持动漫分割和多种 RemoveBG 模型，可输出遮罩和纯色背景预览。
- Added **DvD IOPaint SAM Interactive Segmentation** with SAM ViT-B, ViT-L and
  ViT-H. Left-click includes an area and right-click excludes it.
  / 新增 SAM1 点击选物节点，支持 ViT-B、ViT-L、ViT-H 和前景/背景点。
- Added optional `IMAGE` and `MASK` inputs to the eraser. A connected mask is
  combined with strokes painted on the node.
  / 消除节点新增可选图片和遮罩输入，外部遮罩可与节点手绘遮罩合并。
- Added `mask_expand` and `mask_blur` to SAM, plus `edge_blur` to the eraser.
  / SAM 新增遮罩扩展与边缘模糊，消除节点新增最终边缘融合。
- Added bilingual node descriptions, input tooltips and output tooltips.
  / 为三个节点补充中英文功能简介、参数悬停说明和输出说明。
- Added separate model folders for erase, mask and interactive-segmentation
  models below `ComfyUI/models/iopaint`.
  / 在 `ComfyUI/models/iopaint` 下按消除、抠图和交互分割分类保存模型。

### Changed / 调整

- SAM `auto_run` is disabled by default so several points can be placed before
  running. / SAM 默认关闭自动运行，方便先添加多个点再提交。
- SAM outputs are ordered as `IMAGE` first and `MASK` second.
  / SAM 输出调整为上方 `IMAGE`、下方 `MASK`。
- The SAM canvas now hides ComfyUI's duplicate native image preview and sizes
  itself to the source image. / SAM 画布会隐藏重复的原生预览并自动适配图片比例。
- Valid legacy erase-model files in the old `models/iopaint` root are migrated
  to `models/iopaint/erase` when first checked.
  / 旧目录中可识别且校验正确的消除模型会在首次检查时迁移到 `erase`。

### Fixed / 修复

- Fixed execution states that could remain stuck on `Running` after a failure
  or a prompt without a result. / 修复部分失败或无结果流程一直显示运行中的问题。
- Fixed the comparison view so its before image uses a connected `IMAGE`
  instead of the node's file-picker image. / 修复接入外部图片时对比区仍显示默认图片的问题。
- Fixed prompt validation for linked image inputs in Mask Generator and SAM.
  / 修复自动抠图与 SAM 节点连接图片后可能无法通过工作流校验的问题。
- Improved external mask shape handling and model runtime compatibility.
  / 改进外部遮罩尺寸兼容与多个消除模型的运行兼容性。

### Upgrade notes / 升级提示

- Existing workflows created with an earlier SAM build should reconnect its
  outputs after upgrading because the slot order changed.
  / 早期 SAM 工作流升级后请重新连接输出端口，避免旧连线仍使用原端口序号。
- Mask Generator uses the optional dependencies in `requirements-mask.txt`.
  The eraser and SAM node do not require that file.
  / 自动抠图节点需要安装 `requirements-mask.txt`，消除节点和 SAM 不需要。
- Neural model weights are still downloaded only when first used and are not
  included in the repository. / 神经网络权重仍在首次使用时下载，不包含在仓库中。

## 1.0.0 - 2026-09-02

- Initial release of the interactive eraser with painting, automatic execution,
  continuous editing, undo/restore, image dropping and before/after comparison.
  / 首次发布交互式消除节点，包含手绘、自动运行、连续编辑、撤销/恢复、拖入图片和前后对比。
