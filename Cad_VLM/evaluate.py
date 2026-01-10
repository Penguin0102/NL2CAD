# -*- coding: utf-8 -*-
"""
NL2CAD 模型评估脚本
====================
生成完整的评估报告，包括：
1. 序列准确率统计
2. Chamfer距离评估
3. 几何元素统计（精确率、召回率和F1分数）
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))

import torch
import argparse
import yaml
from tqdm import tqdm
from rich import print
import warnings
import logging.config

warnings.filterwarnings("ignore")
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": True,
})

from CadSeqProc.cad_sequence import CADSequence
from CadSeqProc.utility.macro import *
from CadSeqProc.utility.utils import chamfer_dist, normalize_pc, ensure_dir
from CadSeqProc.utility.logger import CLGLogger
from Cad_VLM.models.text2cad import Text2CAD
from Cad_VLM.models.metrics import AccuracyCalculator
from Cad_VLM.dataprep.t2c_dataset import get_dataloaders

logger = CLGLogger().configure_logger(verbose=True).logger


def parse_config_file(config_file):
    with open(config_file, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data


def evaluate_model(model, test_loader, device, config, output_dir):
    """
    评估模型性能
    
    Returns:
        evaluation_results: 包含所有评估指标的字典
    """
    model.eval()
    
    # 存储评估结果
    all_results = {}
    seq_accuracies = []
    chamfer_distances = []
    valid_count = 0
    invalid_count = 0
    
    # 几何元素统计
    geometry_stats = {
        "line": {"tp": 0, "fp": 0, "fn": 0},
        "arc": {"tp": 0, "fp": 0, "fn": 0},
        "circle": {"tp": 0, "fp": 0, "fn": 0},
        "extrusion": {"tp": 0, "fp": 0, "fn": 0}
    }
    
    TOPK = 5 if config.get("test", {}).get("sampling_type", "topk") != "max" else 1
    
    with torch.no_grad():
        with tqdm(test_loader, ascii=True, desc="评估进度") as pbar:
            for uid_level, vec_dict, prompt, _ in pbar:
                for key, value in vec_dict.items():
                    vec_dict[key] = value.to(device)
                
                for topk_index in range(1, TOPK + 1):
                    try:
                        # 自回归预测
                        pred_cad_seq_dict = model.test_decode(
                            texts=prompt,
                            maxlen=MAX_CAD_SEQUENCE_LENGTH,
                            nucleus_prob=0,
                            topk_index=topk_index,
                            device=device,
                        )
                        
                        # 计算序列准确率
                        seq_acc = AccuracyCalculator(
                            discard_token=len(END_TOKEN)
                        ).calculateAccMulti2DFromLabel(
                            pred_cad_seq_dict["cad_vec"].cpu(),
                            vec_dict["cad_vec"].cpu(),
                        )
                        seq_accuracies.append(seq_acc)
                        
                        # 处理每个样本
                        for i in range(vec_dict["cad_vec"].shape[0]):
                            uid, level = uid_level[i].split("_")
                            
                            if uid not in all_results:
                                all_results[uid] = {}
                            if level not in all_results[uid]:
                                all_results[uid][level] = {
                                    "pred_cad_vec": [],
                                    "gt_cad_vec": None,
                                    "cd": [],
                                    "seq_acc": []
                                }
                            
                            try:
                                # 构建Ground Truth CAD模型
                                gt_cad = (
                                    CADSequence.from_vec(
                                        vec_dict["cad_vec"][i].cpu().numpy(),
                                        bit=N_BIT,
                                        post_processing=True,
                                    )
                                    .create_cad_model()
                                    .sample_points(n_points=8192)
                                )
                                
                                # 构建预测CAD模型
                                pred_cad = (
                                    CADSequence.from_vec(
                                        pred_cad_seq_dict["cad_vec"][i].cpu().numpy(),
                                        bit=N_BIT,
                                        post_processing=True,
                                    )
                                    .create_cad_model()
                                    .sample_points(n_points=8192)
                                )
                                
                                # 计算Chamfer距离
                                cd = chamfer_dist(
                                    normalize_pc(gt_cad.points),
                                    normalize_pc(pred_cad.points),
                                ) * 1000
                                
                                chamfer_distances.append(cd)
                                valid_count += 1
                                
                                all_results[uid][level]["cd"].append(cd)
                                all_results[uid][level]["pred_cad_vec"].append(
                                    pred_cad_seq_dict["cad_vec"][i].cpu().numpy()
                                )
                                all_results[uid][level]["gt_cad_vec"] = vec_dict["cad_vec"][i].cpu().numpy()
                                all_results[uid][level]["seq_acc"].append(seq_acc)
                                
                                # 更新几何元素统计
                                try:
                                    gt_cad_obj = CADSequence.from_vec(
                                        vec_dict["cad_vec"][i].cpu().numpy(),
                                        bit=N_BIT,
                                        post_processing=True,
                                    )
                                    pred_cad_obj = CADSequence.from_vec(
                                        pred_cad_seq_dict["cad_vec"][i].cpu().numpy(),
                                        bit=N_BIT,
                                        post_processing=True,
                                    )
                                    
                                    # 统计几何元素
                                    update_geometry_stats(geometry_stats, gt_cad_obj, pred_cad_obj)
                                except:
                                    pass
                                    
                            except Exception as e:
                                invalid_count += 1
                                all_results[uid][level]["cd"].append(-1)
                                
                            pbar.set_postfix({
                                "valid": valid_count, 
                                "invalid": invalid_count,
                                "seq_acc": f"{np.mean(seq_accuracies):.2f}" if seq_accuracies else "N/A"
                            })
                            
                    except Exception as e:
                        logger.error(f"评估错误: {e}")
                        continue
    
    # 生成评估报告
    evaluation_results = generate_evaluation_report(
        seq_accuracies, chamfer_distances, geometry_stats, 
        valid_count, invalid_count, output_dir
    )
    
    # 保存原始结果
    results_path = os.path.join(output_dir, "evaluation_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"原始评估结果已保存到: {results_path}")
    
    return evaluation_results


def update_geometry_stats(stats, gt_cad, pred_cad):
    """更新几何元素统计"""
    # 统计线条
    gt_lines = sum(1 for skt in gt_cad.sketch_seq for face in skt.facedata 
                   for loop in face.loopdata for curve in loop.curvedata 
                   if curve.curve_type.lower() == "line")
    pred_lines = sum(1 for skt in pred_cad.sketch_seq for face in skt.facedata 
                     for loop in face.loopdata for curve in loop.curvedata 
                     if curve.curve_type.lower() == "line")
    
    # 统计弧线
    gt_arcs = sum(1 for skt in gt_cad.sketch_seq for face in skt.facedata 
                  for loop in face.loopdata for curve in loop.curvedata 
                  if curve.curve_type.lower() == "arc")
    pred_arcs = sum(1 for skt in pred_cad.sketch_seq for face in skt.facedata 
                    for loop in face.loopdata for curve in loop.curvedata 
                    if curve.curve_type.lower() == "arc")
    
    # 统计圆
    gt_circles = sum(1 for skt in gt_cad.sketch_seq for face in skt.facedata 
                     for loop in face.loopdata for curve in loop.curvedata 
                     if curve.curve_type.lower() == "circle")
    pred_circles = sum(1 for skt in pred_cad.sketch_seq for face in skt.facedata 
                       for loop in face.loopdata for curve in loop.curvedata 
                       if curve.curve_type.lower() == "circle")
    
    # 统计挤压操作
    gt_extrusions = len(gt_cad.extrude_seq)
    pred_extrusions = len(pred_cad.extrude_seq)
    
    # 更新统计 (简化版本：匹配数量作为TP)
    stats["line"]["tp"] += min(gt_lines, pred_lines)
    stats["line"]["fp"] += max(0, pred_lines - gt_lines)
    stats["line"]["fn"] += max(0, gt_lines - pred_lines)
    
    stats["arc"]["tp"] += min(gt_arcs, pred_arcs)
    stats["arc"]["fp"] += max(0, pred_arcs - gt_arcs)
    stats["arc"]["fn"] += max(0, gt_arcs - pred_arcs)
    
    stats["circle"]["tp"] += min(gt_circles, pred_circles)
    stats["circle"]["fp"] += max(0, pred_circles - gt_circles)
    stats["circle"]["fn"] += max(0, gt_circles - pred_circles)
    
    stats["extrusion"]["tp"] += min(gt_extrusions, pred_extrusions)
    stats["extrusion"]["fp"] += max(0, pred_extrusions - gt_extrusions)
    stats["extrusion"]["fn"] += max(0, gt_extrusions - pred_extrusions)


def calculate_metrics(tp, fp, fn):
    """计算精确率、召回率和F1分数"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def generate_evaluation_report(seq_accuracies, chamfer_distances, geometry_stats, 
                               valid_count, invalid_count, output_dir):
    """生成完整的评估报告"""
    
    report = {
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        
        # 1. 序列准确率统计
        "sequence_accuracy": {
            "mean": float(np.mean(seq_accuracies)) if seq_accuracies else 0,
            "std": float(np.std(seq_accuracies)) if seq_accuracies else 0,
            "min": float(np.min(seq_accuracies)) if seq_accuracies else 0,
            "max": float(np.max(seq_accuracies)) if seq_accuracies else 0,
            "median": float(np.median(seq_accuracies)) if seq_accuracies else 0,
            "total_samples": len(seq_accuracies)
        },
        
        # 2. Chamfer距离评估
        "chamfer_distance": {
            "mean": float(np.mean(chamfer_distances)) if chamfer_distances else 0,
            "std": float(np.std(chamfer_distances)) if chamfer_distances else 0,
            "median": float(np.median(chamfer_distances)) if chamfer_distances else 0,
            "min": float(np.min(chamfer_distances)) if chamfer_distances else 0,
            "max": float(np.max(chamfer_distances)) if chamfer_distances else 0,
            "unit": "×10^-3"
        },
        
        # 有效性统计
        "validity": {
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "validity_rate": valid_count / (valid_count + invalid_count) * 100 if (valid_count + invalid_count) > 0 else 0
        },
        
        # 3. 几何元素统计
        "geometry_elements": {}
    }
    
    # 计算每种几何元素的精确率、召回率和F1
    for element_type, stats in geometry_stats.items():
        precision, recall, f1 = calculate_metrics(stats["tp"], stats["fp"], stats["fn"])
        report["geometry_elements"][element_type] = {
            "precision": float(precision) * 100,
            "recall": float(recall) * 100,
            "f1_score": float(f1) * 100,
            "true_positives": stats["tp"],
            "false_positives": stats["fp"],
            "false_negatives": stats["fn"]
        }
    
    # 保存JSON报告
    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    # 生成可读的文本报告
    text_report = generate_text_report(report)
    text_report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(text_report_path, "w", encoding="utf-8") as f:
        f.write(text_report)
    
    # 生成CSV报告
    generate_csv_report(report, output_dir)
    
    logger.success(f"评估报告已保存到: {output_dir}")
    print(text_report)
    
    return report


def generate_text_report(report):
    """生成可读的文本报告"""
    lines = [
        "=" * 60,
        "NL2CAD 模型性能评估报告",
        "=" * 60,
        f"评估时间: {report['evaluation_date']}",
        "",
        "-" * 60,
        "4.2.1 序列准确率统计",
        "-" * 60,
        f"  平均准确率: {report['sequence_accuracy']['mean']:.2%}",
        f"  标准差: {report['sequence_accuracy']['std']:.4f}",
        f"  最小值: {report['sequence_accuracy']['min']:.2%}",
        f"  最大值: {report['sequence_accuracy']['max']:.2%}",
        f"  中位数: {report['sequence_accuracy']['median']:.2%}",
        f"  样本数量: {report['sequence_accuracy']['total_samples']}",
        "",
        "-" * 60,
        "4.2.2 Chamfer距离评估",
        "-" * 60,
        f"  平均距离: {report['chamfer_distance']['mean']:.4f} (×10^-3)",
        f"  标准差: {report['chamfer_distance']['std']:.4f}",
        f"  中位数: {report['chamfer_distance']['median']:.4f}",
        f"  最小值: {report['chamfer_distance']['min']:.4f}",
        f"  最大值: {report['chamfer_distance']['max']:.4f}",
        "",
        f"  有效样本数: {report['validity']['valid_count']}",
        f"  无效样本数: {report['validity']['invalid_count']}",
        f"  有效率: {report['validity']['validity_rate']:.2f}%",
        "",
        "-" * 60,
        "4.2.3 几何元素统计",
        "-" * 60,
        "  元素类型      精确率     召回率     F1分数",
        "  " + "-" * 50,
    ]
    
    for element_type, stats in report["geometry_elements"].items():
        lines.append(
            f"  {element_type:<12} {stats['precision']:>8.2f}%  {stats['recall']:>8.2f}%  {stats['f1_score']:>8.2f}%"
        )
    
    lines.extend([
        "",
        "=" * 60,
        "评估完成",
        "=" * 60,
    ])
    
    return "\n".join(lines)


def generate_csv_report(report, output_dir):
    """生成CSV格式的报告"""
    
    # 序列准确率CSV
    seq_df = pd.DataFrame([report["sequence_accuracy"]])
    seq_df.to_csv(os.path.join(output_dir, "sequence_accuracy.csv"), index=False)
    
    # Chamfer距离CSV
    cd_df = pd.DataFrame([report["chamfer_distance"]])
    cd_df.to_csv(os.path.join(output_dir, "chamfer_distance.csv"), index=False)
    
    # 几何元素统计CSV
    geo_data = []
    for element_type, stats in report["geometry_elements"].items():
        geo_data.append({
            "element_type": element_type,
            **stats
        })
    geo_df = pd.DataFrame(geo_data)
    geo_df.to_csv(os.path.join(output_dir, "geometry_elements.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="NL2CAD 模型评估")
    parser.add_argument(
        "-c", "--config_path",
        type=str,
        default="config/inference.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "-m", "--model_path",
        type=str,
        default=None,
        help="模型检查点路径"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default=None,
        help="输出目录"
    )
    args = parser.parse_args()
    
    # 加载配置
    config = parse_config_file(args.config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            config.get("test", {}).get("log_dir", "logs"),
            f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    ensure_dir(output_dir)
    
    # 加载模型
    cad_config = config["cad_decoder"]
    cad_config["cad_seq_len"] = MAX_CAD_SEQUENCE_LENGTH
    model = Text2CAD(
        text_config=config["text_encoder"],
        cad_config=cad_config
    ).to(device)
    
    # 加载检查点
    checkpoint_path = args.model_path or config.get("test", {}).get("checkpoint_path")
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"加载模型检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        pretrained_dict = {}
        for key, value in checkpoint["model_state_dict"].items():
            if key.split(".")[0] == "module":
                pretrained_dict[".".join(key.split(".")[1:])] = value
            else:
                pretrained_dict[key] = value
        
        model.load_state_dict(pretrained_dict, strict=False)
        logger.success("模型加载成功")
    else:
        logger.warning("未找到模型检查点，使用随机初始化的模型")
    
    # 创建测试数据加载器 (优先使用test，如果为空则使用validation)
    test_loader = get_dataloaders(
        cad_seq_dir=config.get("test_data", config.get("train_data", {})).get("cad_seq_dir"),
        prompt_path=config.get("test_data", config.get("train_data", {})).get("prompt_path"),
        split_filepath=config.get("test_data", config.get("train_data", {})).get("split_filepath"),
        subsets=["test"],
        batch_size=config.get("test", config.get("train", {})).get("batch_size", 1),
        num_workers=min(config.get("test", config.get("train", {})).get("num_workers", 4), os.cpu_count()),
        pin_memory=True,
        shuffle=False,
        prefetch_factor=config.get("test", config.get("train", {})).get("prefetch_factor", 2),
        debug=config.get("debug", False),
    )[0]
    
    # 如果测试集为空，尝试使用验证集
    if len(test_loader.dataset) == 0:
        logger.warning("测试集为空，尝试使用验证集进行评估")
        test_loader = get_dataloaders(
            cad_seq_dir=config.get("test_data", config.get("train_data", {})).get("cad_seq_dir"),
            prompt_path=config.get("test_data", config.get("train_data", {})).get("prompt_path"),
            split_filepath=config.get("test_data", config.get("train_data", {})).get("split_filepath"),
            subsets=["validation"],
            batch_size=config.get("test", config.get("train", {})).get("batch_size", 1),
            num_workers=min(config.get("test", config.get("train", {})).get("num_workers", 4), os.cpu_count()),
            pin_memory=True,
            shuffle=False,
            prefetch_factor=config.get("test", config.get("train", {})).get("prefetch_factor", 2),
            debug=config.get("debug", False),
        )[0]
    
    # 运行评估
    logger.info("开始模型评估...")
    evaluation_results = evaluate_model(model, test_loader, device, config, output_dir)
    
    logger.success("评估完成！")
    return evaluation_results


if __name__ == "__main__":
    main()
