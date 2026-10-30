# 班里挑一

一个完全离线的 Windows 多人脸随机抽取工具。使用 Rust、Slint、Media Foundation 和 YuNet（ONNX Runtime CPU）。

## 运行成品

将 `dist/NBPick` 整个文件夹复制到学校电脑，双击 `班里挑一.exe`。不要只复制 exe，必须保留 `assets` 和 DLL 文件。

- 启动后加载本地模型，再全屏显示摄像头；扫描页不响应鼠标操作。
- 默认扫描 3 秒，从最近的有效画面内均匀随机抽取一张人脸。
- 连续 5 秒没有检测到任何人脸时，进入“未检测到人脸”失败页。
- 点击“再抽一次”重新全屏扫描。允许连续抽到同一个人。
- 抽取结果页可切换摄像头、调整扫描时间为 2–6 秒、静音、重试或退出。
- 结果页按 Enter 或空格可再次抽取，Esc 可退出；设置或摄像头列表打开时，Esc 先关闭面板。
- 没有摄像头、未检测到人脸、设备断开时不会生成结果，会自动重连或等待清晰画面。
- 不保存视频、照片、姓名或身份特征。仅在本机 `%LOCALAPPDATA%\NBPick` 保存摄像头 ID、扫描时间和静音设置。

## 设备要求与排错

- Windows 10/11 x64，可用摄像头和系统摄像头权限。
- 推荐支持 AVX2 的 CPU。预编译 ONNX Runtime 在极老的无 AVX2 处理器上可能无法运行。
- 摄像头被其他程序独占时，关闭其他程序或在右上角切换设备。
- 模型文件或运行库缺失会显示错误，重新复制完整发布目录。
- 图形驱动过旧时，设置环境变量 `SLINT_BACKEND=winit-software` 后运行，可强制软件渲染。
- 无声音设备不影响抽取，程序继续运行。

## 开发

```powershell
cargo test
cargo clippy --all-targets -- -D warnings
cargo run --bin nb-pick
```

构建发布目录：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/package.ps1
```

`camera_check` 与 `model_check` 为可选诊断工具。后者可接受一个本地图片路径，检测人数和坐标只输出到终端。

## 许可证

YuNet 模型来源与许可证见 `assets/models/LICENSE-YuNet`。音效由程序实时合成，无外部音频素材。
