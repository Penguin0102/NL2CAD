1. `conda env create -f environment.yaml` 配置环境
2. 从<https://huggingface.co/datasets/SadilKhan/NL2CAD/tree/main/nl2cad_v1.0> 下载NL2CAD_1.0.pth放入trained目录
3. 运行demo

```
cd App
gradio app.py
```

4. 训练

```
python Cad_VLM/train.py --config_path Cad_VLM/config/trainer_user_test.yaml
```
