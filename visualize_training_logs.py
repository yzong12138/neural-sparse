#!/usr/bin/env python3
"""
训练日志可视化脚本
用法: python visualize_training_logs.py path1 path2 path3 ...
"""

import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_log_file(log_path):
    """解析训练日志文件"""
    data = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "ranking loss moving avg" in line:
                # 提取步数和指标
                step_match = re.search(r"Step (\d+)", line)
                loss_match = re.search(r"ranking loss moving avg:([0-9.e-]+)", line)
                flops_match = re.search(r"d_flops: ([0-9.e-]+)", line)
                flops_loss_match = re.search(r"flops_loss: ([0-9.e-]+)", line)
                doc_len_match = re.search(r"avg doc length: ([0-9.]+)", line)

                if all(
                    [
                        step_match,
                        loss_match,
                        flops_match,
                        flops_loss_match,
                        doc_len_match,
                    ]
                ):
                    data.append(
                        {
                            "step": int(step_match.group(1)),
                            "ranking_loss": float(loss_match.group(1)),
                            "d_flops": float(flops_match.group(1)),
                            "flops_loss": float(flops_loss_match.group(1)),
                            "avg_doc_length": float(doc_len_match.group(1)),
                        }
                    )

            elif "nonzero entries:" in line:
                # 提取nonzero entries的三个数值
                numbers = re.findall(
                    r"nonzero entries: ([0-9.e-]+) ([0-9.e-]+) ([0-9.e-]+)", line
                )
                if numbers and data:  # 确保有对应的step数据
                    data[-1]["nonzero_1"] = float(numbers[0][0])
                    data[-1]["nonzero_2"] = float(numbers[0][1])
                    data[-1]["nonzero_3"] = float(numbers[0][2])

    return pd.DataFrame(data)


def aggregate_data(df, step_interval=1000):
    """按指定步数间隔聚合数据"""
    if df.empty:
        return df

    # 创建步数区间
    df["step_group"] = (df["step"] // step_interval) * step_interval

    # 按步数区间分组，取每组的最后一个值（最新的值）
    aggregated = df.groupby("step_group").last().reset_index()

    # 使用step_group作为新的step值
    aggregated["step"] = aggregated["step_group"]
    aggregated = aggregated.drop("step_group", axis=1)

    return aggregated


def get_experiment_name(log_path):
    """从路径中提取实验名称"""
    path = Path(log_path)
    # 尝试从路径中提取有意义的名称
    parts = path.parts
    if len(parts) >= 2:
        return f"{parts[-2]}"  # 使用倒数第二个目录名
    else:
        return path.stem


def create_comparison_plots(log_paths, experiment_names=None, step_interval=1000):
    """创建对比图表"""
    # 准备数据
    all_data = []
    if experiment_names is None:
        experiment_names = [get_experiment_name(path) for path in log_paths]

    for log_path, exp_name in zip(log_paths, experiment_names):
        try:
            df = parse_log_file(log_path)
            if not df.empty:
                # 聚合数据
                df_aggregated = aggregate_data(df, step_interval)
                df_aggregated["experiment"] = exp_name
                all_data.append(df_aggregated)
                print(
                    f"成功解析 {exp_name}: 原始数据 {len(df)} 个点 -> 聚合后 {len(df_aggregated)} 个点 (每{step_interval}步)"
                )
            else:
                print(f"警告: {exp_name} 没有找到有效数据")
        except Exception as e:
            print(f"错误: 无法解析 {log_path}: {e}")

    if not all_data:
        print("错误: 没有找到任何有效数据")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"训练指标对比分析 (每{step_interval}步聚合)", fontsize=16)

    # 设置颜色
    colors = plt.cm.Set1(np.linspace(0, 1, len(experiment_names)))

    # 定义要绘制的指标
    metrics = [
        ("ranking_loss", "ranking loss", (0, 0), False),
        ("d_flops", "d_flops", (0, 1), False),
        ("flops_loss", "flops_loss", (0, 2), True),  # 使用对数刻度
        ("avg_doc_length", "avg_doc_length", (1, 0), False),
        ("nonzero_1", "nonzero_1", (1, 1), False),
        ("nonzero_3", "nonzero_3", (1, 2), False),
    ]

    for metric, title, (row, col), use_log in metrics:
        ax = axes[row, col]

        for i, exp_name in enumerate(experiment_names):
            exp_data = combined_df[combined_df["experiment"] == exp_name]
            if not exp_data.empty and metric in exp_data.columns:
                ax.plot(
                    exp_data["step"],
                    exp_data[metric],
                    label=exp_name,
                    color=colors[i],
                    marker="o",
                    markersize=3,
                    linewidth=2,
                )

        ax.set_xlabel("step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if use_log:
            ax.set_yscale("log")

    plt.tight_layout()

    # 保存图表
    output_path = f"training_metrics_comparison_{step_interval}steps.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"图表已保存到: {output_path}")

    # 显示图表
    plt.show()

    return combined_df


def print_summary_stats(df):
    """打印摘要统计"""
    print("\n=== 训练指标摘要统计 ===")
    metrics = [
        "ranking_loss",
        "d_flops",
        "flops_loss",
        "avg_doc_length",
        "nonzero_1",
        "nonzero_3",
    ]

    for exp_name in df["experiment"].unique():
        exp_data = df[df["experiment"] == exp_name]
        print(f"\n实验: {exp_name}")
        print(f"训练步数范围: {exp_data['step'].min()} - {exp_data['step'].max()}")

        for metric in metrics:
            if metric in exp_data.columns:
                final_value = exp_data[metric].iloc[-1]
                print(f"  {metric} (最终值): {final_value:.6f}")


def main():
    parser = argparse.ArgumentParser(description="可视化训练日志指标对比")
    parser.add_argument("log_paths", nargs="+", help="训练日志文件路径")
    parser.add_argument("--names", nargs="*", help="实验名称 (可选)")
    parser.add_argument(
        "--step-interval", type=int, default=1000, help="聚合步数间隔 (默认: 1000)"
    )

    args = parser.parse_args()

    # 检查文件是否存在
    valid_paths = []
    for path in args.log_paths:
        if Path(path).exists():
            valid_paths.append(path)
        else:
            print(f"警告: 文件不存在 {path}")

    if not valid_paths:
        print("错误: 没有找到有效的日志文件")
        return

    # 创建对比图表
    df = create_comparison_plots(valid_paths, args.names, args.step_interval)

    if df is not None:
        print_summary_stats(df)


if __name__ == "__main__":
    main()
