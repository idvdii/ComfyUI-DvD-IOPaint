# ComfyUI-DvD-IOPaint

[English](README.md) · 简体中文

一个适用于 ComfyUI 的交互式图像消除节点，模型运行时改编自
[IOPaint](https://github.com/Sanster/IOPaint)。

在一个节点内完成图片载入、遮罩涂抹、自动消除、前后对比和连续编辑。

![DvD IOPaint 演示](assets/demo.gif)

## 功能

- 直接在节点画布上涂抹要消除的区域。
- 开启 `auto_run` 后，松开鼠标会自动提交遮罩并运行工作流。
- 处理结果会覆盖画布底图，可以继续进行下一次编辑。
- 将本地图片拖到节点上即可替换当前图片。
- 鼠标位于画布时显示半透明的画笔直径；按住 `Alt` 滚动滚轮可以调节画笔大小。
- 不按 `Alt` 时保留 ComfyUI 原本的节点画布缩放操作。
- 可以撤销尚未处理的遮罩笔画，也可以恢复上一次已经完成的消除结果。
- 在下方对比区移动或拖动鼠标，可以查看处理前后的分界效果。
- 模型只在第一次选用时下载，并进行 MD5 校验。
- 模型文件统一保存到 `ComfyUI/models/iopaint`。
- 不需要安装完整 IOPaint，也不会替换 ComfyUI 自带的 Torch/CUDA 环境。

## 安装

### ComfyUI Manager

选择 **Install via Git URL**，填入：

```text
https://github.com/idvdii/ComfyUI-DvD-IOPaint.git
```

安装完成后重启 ComfyUI。

### 手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/idvdii/ComfyUI-DvD-IOPaint.git
cd ComfyUI-DvD-IOPaint
python -m pip install -r requirements.txt
```

请使用 ComfyUI 对应的 Python。Windows 便携版通常会在 `ComfyUI` 同级目录
提供内置 Python 可执行文件。

## 使用方法

1. 在 `DvD/Image` 分类下添加 **DvD IOPaint Interactive Eraser**。
2. 选择图片，或直接将本地图片拖到节点上。
3. 选择模型，涂抹要消除的物体或文字。
4. 松开鼠标。开启 `auto_run` 时会自动排队运行，结果会成为下一次编辑的底图。

工具栏按钮：

| 按钮 | 作用 |
| --- | --- |
| 铅笔 | 绘制消除遮罩 |
| 橡皮 | 擦除遮罩的一部分 |
| 撤销 | 撤销最近一次尚未处理的遮罩笔画 |
| 历史 | 恢复上一次消除前的图片 |
| 垃圾桶 | 清空当前遮罩 |

节点输出处理后的 `IMAGE` 和本次提交的 `MASK`。

## 模型

文件大小为约数，方便估算磁盘和网络占用。

| 模型 | 下载大小 | 说明 |
| --- | ---: | --- |
| LaMa | 约 196 MiB | 通用大遮罩消除 |
| Anime LaMa | 约 196 MiB | 面向动漫和漫画的 LaMa 权重 |
| AOT Manga/Anime | 约 22 MiB | 轻量彩色动漫/漫画选项 |
| MAT | 约 239 MiB | 注重结构的通用修复 |
| MIGAN | 约 27 MiB | 轻量级 512 尺寸修复 |
| LDM | 约 1.6 GiB，3 个文件 | 基于扩散的消除模型，速度较慢且占用较大 |
| ZITS | 约 600 MiB，4 个文件 | 注重结构和线稿的修复 |
| FcF | 约 327 MiB | 适合较大缺口的频域修复 |
| Manga B&W Semantic | 约 235 MiB，2 个文件 | 仅适合黑白漫画，遮罩区域输出为灰度 |
| OpenCV Telea | 0 MiB | 适合小缺陷的快速传统修复 |
| OpenCV Navier-Stokes | 0 MiB | 适合小缺陷的快速传统修复 |

神经网络模型首次使用时需要访问对应的 GitHub 或 Hugging Face 下载地址。
下载中断或校验失败的文件会被删除，下次运行时自动重新下载。

## 依赖与兼容性

`requirements.txt` 只包含本节点额外需要的包：

- `opencv-contrib-python-headless`：图像处理和 OpenCV 传统修复
- `scikit-image`：ZITS 的边缘和线稿处理

Torch、NumPy、Pillow、Pydantic、SciPy 和 tqdm 由 ComfyUI 提供，因此这里不会
锁定版本。这样可以避免覆盖用户针对显卡配置的 Torch 环境。节点只携带隔离的
IOPaint 消除模型运行时，不会安装完整 IOPaint 应用。

已在 ComfyUI `v0.30.2`、Python 3.13、PyTorch 2.9.1 和 CUDA 环境测试。CPU
也可以运行，但速度较慢。可在 ComfyUI 目录执行冒烟测试：

```bash
python custom_nodes/ComfyUI-DvD-IOPaint/tests/test_smoke.py
```

## 致谢与许可证

消除模型的下载地址、校验值、预处理逻辑和运行时源码改编自 IOPaint 1.6.0。
具体上游提交和改动说明请参见 [NOTICE](NOTICE)。感谢 IOPaint 社区提供开源的
图像修复实现和模型整合工作。

本项目遵循 [Apache License 2.0](LICENSE)。
