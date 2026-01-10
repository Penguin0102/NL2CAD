# -*- coding: utf-8 -*-
"""
NL2CAD 完整训练与评估流程
============================
这个脚本整合了训练和评估的完整流程，生成：
1. 训练日志
2. TensorBoard训练曲线
3. 模型性能评估报告

运行方式:
    conda activate NL2CAD
    python run_full_pipeline.py
    
或者单独运行:
    python run_full_pipeline.py --train-only   # 只训练
    python run_full_pipeline.py --eval-only    # 只评估
"""

import os
import sys
import argparse
import json
import datetime
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def print_header(title):
    """打印分隔线标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def find_latest_checkpoint(log_dir):
    """查找最新的模型检查点"""
    pattern = os.path.join(log_dir, "**", "*.pth")
    checkpoints = glob.glob(pattern, recursive=True)
    if not checkpoints:
        return None
    # 返回最新修改的文件
    return max(checkpoints, key=os.path.getmtime)


def run_training():
    """运行训练流程"""
    print_header("阶段 1: 模型训练")
    
    print("[INFO] 导入训练模块...")
    from Cad_VLM.train import main as train_main
    
    print("[INFO] 开始训练...")
    print("-" * 50)
    
    train_main()
    
    print("-" * 50)
    print("[SUCCESS] 训练完成！")


def run_evaluation(checkpoint_path=None):
    """运行评估流程"""
    print_header("阶段 2: 模型评估")
    
    print("[INFO] 导入评估模块...")
    from Cad_VLM.evaluate import main as eval_main
    
    # 如果没有指定检查点，尝试找最新的
    if checkpoint_path is None:
        log_dir = "c:/Users/86185/Desktop/NL2CAD/logs"
        checkpoint_path = find_latest_checkpoint(log_dir)
        
    if checkpoint_path:
        print(f"[INFO] 使用检查点: {checkpoint_path}")
        sys.argv = ['evaluate.py', '-c', 'Cad_VLM/config/inference.yaml', '-m', checkpoint_path]
    else:
        print("[WARNING] 未找到模型检查点，将使用随机初始化的模型进行评估")
        sys.argv = ['evaluate.py', '-c', 'Cad_VLM/config/inference.yaml']
    
    print("[INFO] 开始评估...")
    print("-" * 50)
    
    eval_main()
    
    print("-" * 50)
    print("[SUCCESS] 评估完成！")


def print_results_summary():
    """打印结果摘要"""
    print_header("结果摘要")
    
    # 查找最新的训练历史
    log_dir = "c:/Users/86185/Desktop/NL2CAD/logs"
    history_files = glob.glob(os.path.join(log_dir, "**", "training_history.json"), recursive=True)
    
    if history_files:
        latest_history = max(history_files, key=os.path.getmtime)
        print(f"[训练历史] {latest_history}")
        with open(latest_history, 'r', encoding='utf-8') as f:
            history = json.load(f)
            if history.get("train_loss"):
                print(f"  - 最终训练损失: {history['train_loss'][-1]:.4f}")
            if history.get("train_seq_acc"):
                print(f"  - 最终训练准确率: {history['train_seq_acc'][-1]:.2%}")
            if history.get("val_seq_acc"):
                print(f"  - 最终验证准确率: {history['val_seq_acc'][-1]:.2%}")
    
    # 查找最新的评估报告
    eval_files = glob.glob(os.path.join(log_dir, "**", "evaluation_report.json"), recursive=True)
    
    if eval_files:
        latest_eval = max(eval_files, key=os.path.getmtime)
        print(f"\n[评估报告] {latest_eval}")
        with open(latest_eval, 'r', encoding='utf-8') as f:
            report = json.load(f)
            seq_acc = report.get("sequence_accuracy", {})
            cd = report.get("chamfer_distance", {})
            validity = report.get("validity", {})
            
            print(f"  - 序列准确率: {seq_acc.get('mean', 0):.2%}")
            print(f"  - Chamfer距离: {cd.get('mean', 0):.4f} (×10^-3)")
            print(f"  - 有效率: {validity.get('validity_rate', 0):.2f}%")
            
            geo = report.get("geometry_elements", {})
            if geo:
                print(f"\n  几何元素统计:")
                for elem, stats in geo.items():
                    print(f"    {elem}: P={stats.get('precision', 0):.1f}%, R={stats.get('recall', 0):.1f}%, F1={stats.get('f1_score', 0):.1f}%")
    
    print("\n" + "-" * 70)
    print("查看完整结果:")
    print(f"  1. TensorBoard: tensorboard --logdir={log_dir}")
    print(f"  2. 日志文件: {log_dir}")
    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="NL2CAD 完整训练与评估流程")
    parser.add_argument("--train-only", action="store_true", help="只运行训练")
    parser.add_argument("--eval-only", action="store_true", help="只运行评估")
    parser.add_argument("--checkpoint", type=str, default=None, help="评估时使用的模型检查点路径")
    args = parser.parse_args()
    
    print_header("NL2CAD 训练与评估系统")
    print(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if args.eval_only:
            # 只运行评估
            run_evaluation(args.checkpoint)
        elif args.train_only:
            # 只运行训练
            run_training()
        else:
            # 完整流程：训练 + 评估
            run_training()
            run_evaluation()
        
        # 打印结果摘要
        print_results_summary()
        
    except Exception as e:
        print(f"\n[ERROR] 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print_header("流程完成")
    print(f"结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0


if __name__ == "__main__":
    exit(main())
