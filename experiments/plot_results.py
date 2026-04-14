import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def plot_ablation_results(json_path: str, output_dir: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    os.makedirs(output_dir, exist_ok=True)
    
    configs = []
    tokens = []
    latencies = []
    no_answer_rates = []
    
    # 简写 Config Name
    for row in data:
        name = row["Config"].replace("A_Baseline", "A(Base)").replace("B_Hetero", "B(+Hetero)").replace("C_Adaptive", "C(+Adapt)").replace("D_CoVe_Full", "D(+CoVe)")
        configs.append(name)
        tokens.append(row["Avg_Tokens"])
        latencies.append(row["Avg_Latency_ms"])
        no_answer_rates.append(row["No_Answer_Rate_Percent"])
        
    # 图 1: Cost vs Latency (双轴柱状图)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    sns.set_theme(style="whitegrid")
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Model Configuration', fontsize=12)
    ax1.set_ylabel('Avg Tokens Cost', color=color1, fontsize=12)
    bars1 = ax1.bar([i - 0.2 for i in range(len(configs))], tokens, width=0.4, color=color1, alpha=0.7, label='Tokens')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('Avg Latency (ms)', color=color2, fontsize=12)  
    bars2 = ax2.bar([i + 0.2 for i in range(len(configs))], latencies, width=0.4, color=color2, alpha=0.7, label='Latency (ms)')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Cost and Latency Comparison across Ablation Models', fontsize=14)
    plt.xticks(range(len(configs)), configs)
    
    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    
    fig.tight_layout()
    plt.savefig(Path(output_dir) / 'ablation_cost_latency.png', dpi=300)
    plt.close()
    
    # 图 2: No-Answer Rate 折线图
    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid")
    plt.plot(configs, no_answer_rates, marker='o', linewidth=2.5, markersize=8, color='tab:orange')
    plt.fill_between(configs, no_answer_rates, alpha=0.2, color='tab:orange')
    plt.title('No-Answer Rate (Safety) Trend', fontsize=14)
    plt.xlabel('Model Configuration', fontsize=12)
    plt.ylabel('No-Answer Rate (%)', fontsize=12)
    
    for i, rate in enumerate(no_answer_rates):
        plt.text(i, rate + 2, f"{rate}%", ha='center', fontweight='bold')
        
    plt.ylim(0, max(max(no_answer_rates) + 20, 100))
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'ablation_safety.png', dpi=300)
    plt.close()
    
    print(f"✅ Plots generated successfully at {output_dir}")

if __name__ == "__main__":
    json_file = "/root/snap/ZJU-RAG-Thesis-Turing/data/results/automated_ablation.json"
    out_path = "/root/snap/ZJU-RAG-Thesis-Turing/paper/zjuthesis/figures"
    plot_ablation_results(json_file, out_path)
