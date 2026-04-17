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
    name_map = {
        "A_Baseline": "A (Baseline)",
        "B_Hetero": "B (+Hetero)",
        "C_Adaptive": "C (+Adaptive)",
        "D_CoVe_Full": "D (+CoVe)"
    }
    
    for item in data:
        cfg = name_map.get(item["Config"], item["Config"])
        configs.append(cfg)
        tokens.append(item["Avg_Tokens"])
        latencies.append(item["Avg_Latency_ms"])
        no_answer_rates.append(item.get("No_Answer_Rate_Percent", 0.0))

    sns.set_theme(style="whitegrid")

    # 1. Plot Cost vs Latency
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Model Configuration', fontsize=12)
    ax1.set_ylabel('Average Tokens per Query', color=color1, fontsize=12)
    bars = ax1.bar(configs, tokens, color=color1, alpha=0.7, label='Tokens')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Average Latency (ms)', color=color2, fontsize=12)
    line = ax2.plot(configs, latencies, color=color2, marker='o', linewidth=2, label='Latency')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{int(height)}', ha='center', va='bottom')
                
    # Add values on lines
    for i, v in enumerate(latencies):
        ax2.text(i, v + 20, f'{int(v)}ms', ha='center', va='bottom', color=color2, fontweight='bold')

    plt.title('Cost and Latency Comparison Across Ablations (N=7405)', fontsize=14, pad=15)
    fig.tight_layout()
    plt.savefig(Path(output_dir) / 'ablation_cost_latency.png', dpi=300)
    plt.close()

    # 2. Plot Safety (No-Answer Rate)
    plt.figure(figsize=(8, 5))
    colors = sns.color_palette("rocket", len(configs))
    plt.bar(configs, no_answer_rates, color=colors)
    plt.title('Safety Mechanism: No-Answer Rate (N=7405)', fontsize=14, pad=15)
    plt.xlabel('Model Configuration', fontsize=12)
    plt.ylabel('No-Answer Rate (%)', fontsize=12)
    
    for i, rate in enumerate(no_answer_rates):
        plt.text(i, rate + 1, f"{rate:.1f}%", ha='center', fontweight='bold')
        
    plt.ylim(0, max(max(no_answer_rates) + 15, 100))
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'ablation_safety.png', dpi=300)
    plt.close()
    
    print(f"✅ Plots generated successfully at {output_dir}")

if __name__ == "__main__":
    json_file = "data/results/automated_ablation.json"
    out_path = "paper/zjuthesis/figures"
    plot_ablation_results(json_file, out_path)
