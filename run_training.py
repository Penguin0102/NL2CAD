# -*- coding: utf-8 -*-
"""
NL2CAD 完整训练和评估流程
=====================================
生成完整的训练日志、TensorBoard曲线和评估报告

运行方式: 
    conda activate NL2CAD
    python run_training.py
"""

import os
import sys
import json
import datetime
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from Cad_VLM.train import main as train_main
from CadSeqProc.utility.utils import ensure_dir

def generate_training_report(log_dir, config):
    """
    生成训练报告
    """
    report = {
        "training_info": {
            "date": str(datetime.date.today()),
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "config": config
        },
        "model_config": {
            "text_encoder": config.get("text_encoder", {}),
            "cad_decoder": config.get("cad_decoder", {})
        },
        "training_params": config.get("train", {})
    }
    
    report_path = os.path.join(log_dir, "training_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    print(f"[INFO] 训练报告已保存到: {report_path}")
    return report_path


def main():
    print("=" * 60)
    print("NL2CAD 训练与评估系统")
    print("=" * 60)
    
    # 直接调用训练主函数
    print("\n[1/2] 开始模型训练...")
    print("-" * 40)
    
    # 运行训练
    train_main()
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print("\n请查看以下内容:")
    print("1. 训练日志: logs/ 目录下的相应文件")
    print("2. TensorBoard曲线: 运行 tensorboard --logdir=logs")
    print("3. 模型检查点: logs/ 目录下的 .pth 文件")


if __name__ == "__main__":
    main()
