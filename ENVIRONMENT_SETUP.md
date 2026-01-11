# NL2CAD 环境配置指南

本文档详细说明了运行 NL2CAD 程序所需的软硬件环境配置及安装步骤。

## 1. 硬件环境要求

为了确保程序能够流畅运行（特别是训练和推理过程），建议配置如下：

* **操作系统**: Windows 10/11 或 Ubuntu 20.04/22.04 LTS
* **显卡 (GPU)**: NVIDIA 显卡，显存建议 **12GB** 及以上（支持 CUDA 11.8）
  * *最低要求*: NVIDIA GPU 8GB 显存
* **处理器 (CPU)**: Intel Core i7 或 AMD Ryzen 7 及以上
* **内存 (RAM)**: **16GB** 或更高
* **硬盘空间**: 建议预留 **50GB** 以上的可用空间（用于存储数据集、模型权重及环境）

## 2. 软件环境要求

* **Python 版本**: 3.10
* **CUDA 版本**: 11.8
* **包管理工具**: Anaconda 或 Miniconda

## 3. 安装与配置步骤

### 3.1 创建 Conda 环境

在项目根目录下打开终端（或 Anaconda Prompt），运行以下命令根据 `environment.yaml` 文件创建环境：

```bash
conda env create -f environment.yaml
```

激活环境：

```bash
conda activate NL2CAD
```

### 3.2 手动安装关键库 (如果需要)

如果 `environment.yaml` 安装过程中出现问题，可以手动安装核心深度学习库：

```bash
# 安装 PyTorch, torchvision, torchaudio (CUDA 11.8 版本)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118

# 安装 Gradio (用于运行 Demo)
pip install gradio==6.2.0

# 安装 Transformers
pip install transformers==4.26.1
```



## 4. 运行与验证

### 4.1 运行 Web Demo

进入 `App` 目录并启动 Gradio 应用：

```bash
cd App
gradio app.py
```

启动后，在浏览器中访问显示的本地 URL（通常是 `http://127.0.0.1:7860`）。

### 4.2 运行训练脚本

使用以下命令验证训练环境是否配置正确：

```bash
python Cad_VLM/train.py --config_path Cad_VLM/config/trainer_user_test.yaml
```

## 5. 常见问题 (FAQ)

* **显存不足 (OOM)**: 如果在运行过程中出现 `CUDA out of memory`，请尝试减小训练配置中的 `batch_size`。
* **pythonocc 安装失败**: 该项目依赖 `pythonocc-core`，建议通过 `conda` 安装以确保依赖项（如 `occt`）正确配置。
