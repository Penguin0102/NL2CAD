# -*- coding: utf-8 -*-
"""
NL2CAD 完整报告生成脚本
========================
生成训练日志、评估报告和可视化图表

输出到: C:/Users/86185/Desktop/NL2CAD/repot
"""

import os
import sys
import json
import glob
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 非交互式后端
import subprocess
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

REPORT_DIR = r"C:\Users\86185\Desktop\NL2CAD\repot"


def ensure_dir(path):
    """确保目录存在"""
    if not os.path.exists(path):
        os.makedirs(path)


def run_plot_subprocess(task, input_file, output_dir):
    """运行独立的子进程来生成图表"""
    # 确保使用正确的 Python 解释器
    cmd = [sys.executable, __file__, "--task", task, "--input", input_file, "--output", output_dir]
    try:
        # print(f"[INFO] Starting subprocess for task: {task}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            if result.stdout:
                print(result.stdout.strip())
        else:
            print(f"[ERROR] Subprocess for {task} failed with exit code {result.returncode}")
            # if result.stderr:
            #     print(f"[DEBUG] Stderr: {result.stderr.strip()}")
    except Exception as e:
        print(f"[CRITICAL] Failed to run subprocess for {task}: {e}")


def generate_training_curves(history_file, output_dir):
    """
    生成TensorBoard风格的训练曲线
    """
    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    epochs = history.get("epochs", [])
    train_loss = history.get("train_loss", [])
    train_seq_loss = history.get("train_seq_loss", [])
    train_seq_acc = history.get("train_seq_acc", [])
    val_seq_acc = history.get("val_seq_acc", [])
    learning_rate = history.get("learning_rate", [])
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表 - 损失函数曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 损失函数曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_loss, 'b-', linewidth=2, marker='o', label='Total Loss')
    ax1.plot(epochs, train_seq_loss, 'r--', linewidth=2, marker='s', label='Seq Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Function Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # 2. 准确率曲线
    ax2 = axes[0, 1]
    ax2.plot(epochs, [x * 100 for x in train_seq_acc], 'g-', linewidth=2, marker='o', label='Train Acc')
    ax2.plot(epochs, [x * 100 for x in val_seq_acc], 'm--', linewidth=2, marker='s', label='Val Acc')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(0, 100)
    
    # 3. 学习率曲线
    ax3 = axes[1, 0]
    ax3.plot(epochs, learning_rate, 'c-', linewidth=2, marker='o')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=0)
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 4. 训练摘要
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""
    Training Summary
    ================
    
    Total Epochs: {len(epochs)}
    Final Train Loss: {train_loss[-1]:.4f}
    Final Train Acc: {train_seq_acc[-1]*100:.2f}%
    Final Val Acc: {val_seq_acc[-1]*100:.2f}%
    Best Val Acc: {max(val_seq_acc)*100:.2f}%
    Initial LR: {learning_rate[0]:.6f}
    Final LR: {learning_rate[-1]:.6f}
    """
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    try:
        # 保存图表
        curves_path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(curves_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SUCCESS] Training curves saved to: {curves_path}")
        return curves_path
    except Exception as e:
        print(f"[ERROR] Failed to generate training curves: {e}")
        plt.close()
        return None


def generate_evaluation_charts(report_file, output_dir):
    """
    生成评估结果图表
    """
    with open(report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 序列准确率统计
    ax1 = axes[0, 0]
    seq_acc = report.get("sequence_accuracy", {})
    metrics = ['Mean', 'Median', 'Min', 'Max']
    values = [
        seq_acc.get('mean', 0) * 100,
        seq_acc.get('median', 0) * 100,
        seq_acc.get('min', 0) * 100,
        seq_acc.get('max', 0) * 100
    ]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    bars = ax1.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('4.2.1 Sequence Accuracy Statistics', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Chamfer距离评估
    ax2 = axes[0, 1]
    cd = report.get("chamfer_distance", {})
    cd_metrics = ['Mean', 'Median', 'Min', 'Max']
    cd_values = [
        cd.get('mean', 0),
        cd.get('median', 0),
        cd.get('min', 0),
        cd.get('max', 0)
    ]
    colors2 = ['#e67e22', '#1abc9c', '#34495e', '#f39c12']
    bars2 = ax2.bar(cd_metrics, cd_values, color=colors2, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Chamfer Distance (x10^-3)', fontsize=12)
    ax2.set_title('4.2.2 Chamfer Distance Evaluation', fontsize=14, fontweight='bold')
    for bar, val in zip(bars2, cd_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 几何元素统计 - 精确率/召回率/F1
    ax3 = axes[1, 0]
    geo = report.get("geometry_elements", {})
    elements = list(geo.keys())
    x = np.arange(len(elements))
    width = 0.25
    
    precision = [geo[e].get('precision', 0) for e in elements]
    recall = [geo[e].get('recall', 0) for e in elements]
    f1 = [geo[e].get('f1_score', 0) for e in elements]
    
    bars1 = ax3.bar(x - width, precision, width, label='Precision', color='#3498db', edgecolor='black')
    bars2 = ax3.bar(x, recall, width, label='Recall', color='#2ecc71', edgecolor='black')
    bars3 = ax3.bar(x + width, f1, width, label='F1 Score', color='#e74c3c', edgecolor='black')
    
    ax3.set_ylabel('Score (%)', fontsize=12)
    ax3.set_title('4.2.3 Geometry Elements Statistics', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([e.capitalize() for e in elements])
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 有效性统计饼图
    ax4 = axes[1, 1]
    validity = report.get("validity", {})
    valid = validity.get('valid_count', 0)
    invalid = validity.get('invalid_count', 0)
    
    if valid + invalid > 0:
        sizes = [valid, invalid]
        labels = [f'Valid\n({valid})', f'Invalid\n({invalid})']
        colors_pie = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0)
        ax4.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                autopct='%1.1f%%', shadow=True, startangle=90)
    else:
        ax4.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
    ax4.set_title('Model Validity Rate', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    try:
        # 保存图表
        eval_path = os.path.join(output_dir, "evaluation_charts.png")
        plt.savefig(eval_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SUCCESS] Evaluation charts saved to: {eval_path}")
        return eval_path
    except Exception as e:
        print(f"[ERROR] Failed to generate evaluation charts: {e}")
        plt.close()
        return None


def plot_loss_curve(history_file, output_dir):
    """生成独立的损失函数曲线"""
    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    epochs = history.get("epochs", [])
    train_loss = history.get("train_loss", [])
    train_seq_loss = history.get("train_seq_loss", [])
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', linewidth=2, marker='o', label='Total Loss')
    plt.plot(epochs, train_seq_loss, 'r--', linewidth=2, marker='s', label='Seq Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    
    path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Loss curve saved to: {path}")
    return path


def plot_accuracy_curve(history_file, output_dir):
    """生成独立的准确率曲线"""
    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    epochs = history.get("epochs", [])
    train_seq_acc = history.get("train_seq_acc", [])
    val_seq_acc = history.get("val_seq_acc", [])
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [x * 100 for x in train_seq_acc], 'g-', linewidth=2, marker='o', label='Train Acc')
    plt.plot(epochs, [x * 100 for x in val_seq_acc], 'm--', linewidth=2, marker='s', label='Val Acc')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 100)
    
    path = os.path.join(output_dir, "accuracy_curve.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Accuracy curve saved to: {path}")
    return path


def plot_lr_curve(history_file, output_dir):
    """生成独立的学习率曲线"""
    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    epochs = history.get("epochs", [])
    learning_rate = history.get("learning_rate", [])
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rate, 'c-', linewidth=2, marker='o')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    path = os.path.join(output_dir, "lr_curve.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] LR curve saved to: {path}")
    return path


def plot_geometry_details(report_file, output_dir):
    """生成详细的几何元素统计图"""
    with open(report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    geo = report.get("geometry_elements", {})
    elements = list(geo.keys())
    if not elements:
        return None
        
    x = np.arange(len(elements))
    width = 0.25
    
    precision = [geo[e].get('precision', 0) for e in elements]
    recall = [geo[e].get('recall', 0) for e in elements]
    f1 = [geo[e].get('f1_score', 0) for e in elements]
    
    plt.figure(figsize=(12, 7))
    plt.bar(x - width, precision, width, label='Precision', color='#3498db', edgecolor='black')
    plt.bar(x, recall, width, label='Recall', color='#2ecc71', edgecolor='black')
    plt.bar(x + width, f1, width, label='F1 Score', color='#e74c3c', edgecolor='black')
    
    plt.ylabel('Score (%)', fontsize=12)
    plt.title('Detailed Geometry Elements Performance', fontsize=14, fontweight='bold')
    plt.xticks(x, [e.capitalize() for e in elements])
    plt.legend(loc='upper right')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    
    path = os.path.join(output_dir, "geometry_details.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Geometry details saved to: {path}")
    return path


def plot_validity_pie(report_file, output_dir):
    """生成独立的有效性统计饼图"""
    with open(report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)
        
    validity = report.get("validity", {})
    valid = validity.get('valid_count', 0)
    invalid = validity.get('invalid_count', 0)
    
    plt.figure(figsize=(8, 8))
    if valid + invalid > 0:
        sizes = [valid, invalid]
        labels = [f'Valid\n({valid})', f'Invalid\n({invalid})']
        colors_pie = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0)
        plt.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12})
    else:
        plt.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=14)
        
    plt.title('Model Validity Distribution', fontsize=16, fontweight='bold')
    
    path = os.path.join(output_dir, "validity_pie.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Validity pie chart saved to: {path}")
    return path


def plot_training_dashboard(history_file, output_dir):
    """生成综合训练仪表盘"""
    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    epochs = history.get("epochs", [])
    train_loss = history.get("train_loss", [])
    train_seq_acc = history.get("train_seq_acc", [])
    val_seq_acc = history.get("val_seq_acc", [])
    learning_rate = history.get("learning_rate", [])
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
    
    # 1. Loss Detail
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_loss, color='#e74c3c', linewidth=2, marker='o', markersize=4, label='Loss')
    ax1.set_title('Training Loss Evolution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # 2. Accuracy Detail
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, [x*100 for x in train_seq_acc], color='#2ecc71', linewidth=2, marker='o', markersize=4, label='Train')
    ax2.plot(epochs, [x*100 for x in val_seq_acc], color='#3498db', linewidth=2, marker='s', markersize=4, label='Val')
    ax2.set_title('Accuracy Progress (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # 3. Learning Rate
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, learning_rate, color='#f1c40f', linewidth=2)
    ax3.set_title('Learning Rate Decay', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('LR')
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Summary Table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    stats = [
        ['Metric', 'Value'],
        ['Final Loss', f'{train_loss[-1]:.4f}'],
        ['Final Train Acc', f'{train_seq_acc[-1]*100:.2f}%'],
        ['Final Val Acc', f'{val_seq_acc[-1]*100:.2f}%'],
        ['Best Val Acc', f'{max(val_seq_acc)*100:.2f}%'],
        ['Total Epochs', f'{len(epochs)}']
    ]
    table = ax4.table(cellText=stats, loc='center', cellLoc='center', colWidths=[0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#34495e')
    
    plt.suptitle('NL2CAD Training Performance Dashboard', fontsize=20, fontweight='bold', y=0.95)
    
    path = os.path.join(output_dir, "training_dashboard.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Training dashboard saved to: {path}")
    return path


def generate_text_report(history_file, eval_report_file, output_dir):
    """
    生成完整的文本报告
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("4. 四、程序运行结果展示")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # 4.1 训练过程结果
    report_lines.append("4.1 训练过程结果")
    report_lines.append("-" * 50)
    
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        report_lines.append("")
        report_lines.append("4.1.1 训练日志")
        report_lines.append("-" * 30)
        for i, epoch in enumerate(history.get("epochs", [])):
            report_lines.append(f"  Epoch {epoch}:")
            report_lines.append(f"    - Train Loss: {history['train_loss'][i]:.4f}")
            report_lines.append(f"    - Train Seq Acc: {history['train_seq_acc'][i]*100:.2f}%")
            report_lines.append(f"    - Val Seq Acc: {history['val_seq_acc'][i]*100:.2f}%")
            report_lines.append(f"    - Learning Rate: {history['learning_rate'][i]:.6f}")
        
        report_lines.append("")
        report_lines.append("4.1.2 TensorBoard训练曲线")
        report_lines.append("-" * 30)
        report_lines.append("  训练过程中的关键指标变化趋势：")
        report_lines.append("")
        report_lines.append("  损失函数曲线：")
        report_lines.append(f"    - 初始损失: {history['train_loss'][0]:.4f}")
        report_lines.append(f"    - 最终损失: {history['train_loss'][-1]:.4f}")
        report_lines.append(f"    - 损失下降: {(1 - history['train_loss'][-1]/history['train_loss'][0])*100:.2f}%")
        report_lines.append("")
        report_lines.append("  准确率曲线：")
        report_lines.append(f"    - 初始训练准确率: {history['train_seq_acc'][0]*100:.2f}%")
        report_lines.append(f"    - 最终训练准确率: {history['train_seq_acc'][-1]*100:.2f}%")
        report_lines.append(f"    - 最终验证准确率: {history['val_seq_acc'][-1]*100:.2f}%")
    
    report_lines.append("")
    report_lines.append("")
    
    # 4.2 模型性能评估结果
    report_lines.append("4.2 模型性能评估结果")
    report_lines.append("-" * 50)
    
    if os.path.exists(eval_report_file):
        with open(eval_report_file, 'r', encoding='utf-8') as f:
            eval_report = json.load(f)
        
        # 4.2.1 序列准确率统计
        report_lines.append("")
        report_lines.append("4.2.1 序列准确率统计")
        report_lines.append("-" * 30)
        report_lines.append("  在测试集上的评估结果：")
        seq_acc = eval_report.get("sequence_accuracy", {})
        report_lines.append(f"    - 平均准确率: {seq_acc.get('mean', 0)*100:.2f}%")
        report_lines.append(f"    - 标准差: {seq_acc.get('std', 0):.4f}")
        report_lines.append(f"    - 最小值: {seq_acc.get('min', 0)*100:.2f}%")
        report_lines.append(f"    - 最大值: {seq_acc.get('max', 0)*100:.2f}%")
        report_lines.append(f"    - 中位数: {seq_acc.get('median', 0)*100:.2f}%")
        report_lines.append(f"    - 样本数量: {seq_acc.get('total_samples', 0)}")
        
        # 4.2.2 Chamfer距离评估
        report_lines.append("")
        report_lines.append("4.2.2 Chamfer距离评估")
        report_lines.append("-" * 30)
        report_lines.append("  几何形状相似度评估（单位：×10^-3）：")
        cd = eval_report.get("chamfer_distance", {})
        report_lines.append(f"    - 平均距离: {cd.get('mean', 0):.4f}")
        report_lines.append(f"    - 标准差: {cd.get('std', 0):.4f}")
        report_lines.append(f"    - 中位数: {cd.get('median', 0):.4f}")
        report_lines.append(f"    - 最小值: {cd.get('min', 0):.4f}")
        report_lines.append(f"    - 最大值: {cd.get('max', 0):.4f}")
        report_lines.append("")
        report_lines.append("  说明：")
        validity = eval_report.get("validity", {})
        report_lines.append(f"    - 有效样本数: {validity.get('valid_count', 0)}")
        report_lines.append(f"    - 无效样本数: {validity.get('invalid_count', 0)}")
        report_lines.append(f"    - 有效率: {validity.get('validity_rate', 0):.2f}%")
        
        # 4.2.3 几何元素统计
        report_lines.append("")
        report_lines.append("4.2.3 几何元素统计")
        report_lines.append("-" * 30)
        report_lines.append("  对不同几何元素的精确率、召回率和F1分数：")
        report_lines.append("")
        report_lines.append("  元素类型      精确率     召回率     F1分数")
        report_lines.append("  " + "-" * 50)
        
        geo = eval_report.get("geometry_elements", {})
        for element_type, stats in geo.items():
            report_lines.append(
                f"  {element_type:<12} {stats.get('precision', 0):>8.2f}%  {stats.get('recall', 0):>8.2f}%  {stats.get('f1_score', 0):>8.2f}%"
            )
    
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)
    
    # 保存报告
    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "full_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"[SUCCESS] Full report saved to: {report_path}")
    print("\n" + report_text)
    
    return report_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="NL2CAD Report Generator")
    parser.add_argument("--task", type=str, help="Specific plotting task to run")
    parser.add_argument("--input", type=str, help="Input JSON file path")
    parser.add_argument("--output", type=str, help="Output directory")
    args = parser.parse_args()

    # 如果指定了任务，则运行特定绘图任务并退出
    if args.task:
        if args.task == "training_summary":
            generate_training_curves(args.input, args.output)
        elif args.task == "loss":
            plot_loss_curve(args.input, args.output)
        elif args.task == "accuracy":
            plot_accuracy_curve(args.input, args.output)
        elif args.task == "lr":
            plot_lr_curve(args.input, args.output)
        elif args.task == "eval_summary":
            generate_evaluation_charts(args.input, args.output)
        elif args.task == "geometry":
            plot_geometry_details(args.input, args.output)
        elif args.task == "validity":
            plot_validity_pie(args.input, args.output)
        elif args.task == "dashboard":
            plot_training_dashboard(args.input, args.output)
        return

    ensure_dir(REPORT_DIR)
    
    print("=" * 60)
    print("NL2CAD Report Generator")
    print("=" * 60)
    
    # 查找最新的训练历史文件
    history_files = glob.glob(os.path.join(REPORT_DIR, "**", "training_history.json"), recursive=True)
    # 同时在 logs 目录下查找
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    if os.path.exists(LOGS_DIR):
        history_files.extend(glob.glob(os.path.join(LOGS_DIR, "**", "training_history.json"), recursive=True))
    
    eval_files = glob.glob(os.path.join(REPORT_DIR, "**", "evaluation_report.json"), recursive=True)
    if os.path.exists(LOGS_DIR):
        eval_files.extend(glob.glob(os.path.join(LOGS_DIR, "**", "evaluation_report.json"), recursive=True))
    
    history_file = max(history_files, key=os.path.getmtime) if history_files else None
    eval_file = max(eval_files, key=os.path.getmtime) if eval_files else None
    
    if history_file:
        print(f"\n[INFO] Found training history: {history_file}")
        # 运行所有训练相关的绘图任务
        tasks = ["training_summary", "loss", "accuracy", "lr", "dashboard"]
        for task in tasks:
            run_plot_subprocess(task, history_file, REPORT_DIR)
    else:
        print("[WARNING] No training history found")
    
    if eval_file:
        print(f"\n[INFO] Found evaluation report: {eval_file}")
        # 运行所有评估相关的绘图任务
        run_plot_subprocess("eval_summary", eval_file, REPORT_DIR)
        run_plot_subprocess("geometry", eval_file, REPORT_DIR)
        run_plot_subprocess("validity", eval_file, REPORT_DIR)
    else:
        print("[WARNING] No evaluation report found")
    
    # 生成完整文本报告
    generate_text_report(
        history_file or "", 
        eval_file or "", 
        REPORT_DIR
    )
    
    print("\n" + "=" * 60)
    print(f"All reports saved to: {REPORT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
