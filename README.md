1. `conda env create -f environment.yaml` 配置环境
2. 将模型训练好的pth放入trained目录
3. 运行demo

```
cd App
gradio app.py
```

1. 训练

```
python Cad_VLM/train.py --config_path Cad_VLM/config/trainer.yaml
```
