# 模型文件不在本仓库里

`models/` 目录需要下面 4 个 ONNX 文件，**没有它们程序无法启动**。
它们没有放进 Git 仓库，原因是体积：合计 189MB，其中 `w600k_r50.onnx` 单文件 167MB，
超过 GitHub 单文件 100MB 的硬限制（用 Git LFS 才能放，但克隆方也必须装 LFS，容易踩坑）。

| 文件名 | 大小 | 用途 | 来源 |
|---|---|---|---|
| `det_10g.onnx` | 17MB | 人脸检测（SCRFD） | InsightFace 官方模型包 **buffalo_l** |
| `w600k_r50.onnx` | 167MB | 人脸特征提取（ArcFace，512 维） | 同上 |
| `2d106det.onnx` | 4.8MB | 106 点关键点（定位嘴部区域） | 同上 |
| `minifasnet_v2.onnx` | 1.7MB | 静默活体检测（拦照片/屏幕翻拍） | MiniFASNetV2 的 ONNX 版本 |

## 最简单的办法：从原电脑拷过来

如果你就是本项目的作者、只是换台电脑，**直接把整个 `models/` 目录拷贝过去**即可，
这是最稳妥的方式（文件一致性有保证）。

## 从网络获取

1. **InsightFace 三个模型**：下载官方 `buffalo_l` 模型包（<https://github.com/deepinsight/insightface> 的
   model_zoo 页有下载链接），解压后从里面取出
   `det_10g.onnx`、`w600k_r50.onnx`、`2d106det.onnx` 三个文件，放进本目录。
   也可以用 Python 包自动下载：`pip install insightface` 后首次运行会落到
   `~/.insightface/models/buffalo_l/`，把那三个文件复制过来即可。
2. **minifasnet_v2.onnx**：由开源项目 Silent-Face-Anti-Spoofing 的 MiniFASNetV2 权重导出为 ONNX。
   注意本项目的活体输出是 **3 类 `[打印攻击, 真人, 翻拍攻击]`**，
   代码里真人固定取索引 1（见 `backend/recognizer.py` 的 `LIVENESS_LIVE_INDEX`）。
   换成别的活体模型时，这个索引和输入尺度（`LIVENESS_INPUT_SCALE`）都要核对，否则会判反。

## 放好之后自检

```bash
python -c "import sys; sys.path.insert(0,'.'); from backend.recognizer import recognizer; print(recognizer.has_liveness(), recognizer.has_landmark())"
```

启动服务后也可以看 `/api/health`，`model_ready`、`liveness_ready`、`landmark_ready` 三个字段都应为 `true`。
