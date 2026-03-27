#  OpenVINO + PaddleOCR-VL 文档理解打卡任务

> 飞桨黑客松第十期 · OpenVINO 赛道热身打卡  
> 成功运行 [PaddleOCR-VL Notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/paddleocr_vl) 并完成端到端推理验证 ✅

---

## 📌 任务目标

本仓库用于记录我在 **飞桨黑客松第十期 OpenVINO 赛道** 中完成的热身打卡任务。通过本次实践，我掌握了：

- 如何在 Intel 平台上部署并运行 OpenVINO 官方提供的 PaddleOCR-VL Notebook；
- 理解模型下载 → OpenVINO 转换 → 压缩优化 → 推理执行的完整流程；
- 验证多模态文档理解（OCR）结果的准确性与性能表现。



## 🔧 环境信息

| 项目             | 配置                                                                 |
|------------------|----------------------------------------------------------------------|
| 操作系统         | Ubuntu 20.04.6 LTS (Focal Fossa)                                     |
| CPU              | Intel(R) Xeon(R) Platinum 8350C CPU @ 2.60GHz                        |
| GPU              | NVIDIA Tesla V100-SXM2-32GB                                          |
| GPU 驱动版本     | 570.148.08                                                           |
| CUDA 版本        | 12.8                                                                 |
| OpenVINO 版本    | 2026.0.0-20965-c6d6a13a886-releases/2026/0                           |
| Python 版本      | 3.10.10                                                              |
| 运行平台         | 百度飞桨 AI Studio                                                   |


本仓库包含三个核心文件：
- environment.ipynb:配置初始环境
- paddleocr_vl.ipynb：基于 OpenVINO 的 PaddleOCR-VL 文档理解 Notebook，展示模型加载、转换与推理全过程。
- app_gradio.py：使用 Gradio 构建的轻量 Web 应用，支持上传图片并实时返回 OCR 识别结果。
