import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_ablation_results():
    """
    Reads the automated ablation results and generates two charts:
    1. Token Cost vs Latency (Dual-axis bar chart)
    2. No-Answer Rate Trend (Line chart)
    """
    results_file = Path(__file__).resolve().parent.parent / "data" / "results" / "real_ablation.json"
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return
        
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    models = [d["Config"] for d in data]
    display_labels = [
        label.replace("A2 (Base+Adaptive)", "A2")
        .replace("A3 (Base+CoVe)", "A3")
        .replace("A (Baseline)", "A")
        .replace("B (Hetero)", "B")
        .replace("C (Hetero+Adaptive)", "C")
        .replace("D (CoVe Full)", "D")
        for label in models
    ]
    tokens = [d["Avg_Tokens"] for d in data]
    latencies = [d["Avg_Latency_ms"] for d in data]
    no_answer_rates = [d["No_Answer_Rate_Percent"] for d in data]
    f1_scores = [d.get("F1_Score", 0) for d in data]
    
    # Output directories
    out_dirs = [
        Path(__file__).resolve().parent.parent / "docs" / "images",
        Path(__file__).resolve().parent.parent / "paper" / "zjuthesis" / "figures",
    ]
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # -----------------------------------------
    # Chart 1: Tokens vs Latency (Dual Axis)
    # -----------------------------------------
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    color1 = '#2ca02c' # Green for Tokens
    color2 = '#d62728' # Red for Latency
    
    rects1 = ax1.bar(x - width/2, tokens, width, label='Avg Tokens', color=color1)
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, latencies, width, label='Avg Latency (ms)', color=color2)
    
    # Labels and Titles
    ax1.set_ylabel('Token Cost per Query', color=color1, fontsize=12, fontweight='bold')
    ax2.set_ylabel('Latency (ms)', color=color2, fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_labels, fontsize=11)
    
    # Tick colors
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Cost and Latency Trade-off Across Ablation Models (N=500)', fontsize=14, pad=15)
    fig.tight_layout()
    
    for out_dir in out_dirs:
        chart1_path = out_dir / "ablation_cost_latency.png"
        plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # -----------------------------------------
    # Chart 2: No-Answer Rate and F1 Score Trend
    # -----------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color_f1 = '#8c564b' # Brown for F1
    color_na = '#1f77b4' # Blue for No-Answer
    
    # Plot No-Answer Rate
    ax1.plot(display_labels, no_answer_rates, marker='o', markersize=10, linewidth=3, color=color_na, label='No-Answer Rate (%)')
    for i, txt in enumerate(no_answer_rates):
        ax1.annotate(f"{txt}%", (x[i], no_answer_rates[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=11, fontweight='bold', color=color_na)
    
    # Plot F1 Score
    ax1.plot(display_labels, f1_scores, marker='s', markersize=10, linewidth=3, color=color_f1, linestyle='--', label='F1 Score')
    for i, txt in enumerate(f1_scores):
        ax1.annotate(f"{txt}", (x[i], f1_scores[i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=11, fontweight='bold', color=color_f1)
        
    ax1.set_ylabel('Score / Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Safety Mechanism vs Model Accuracy (F1 Score)', fontsize=14, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_labels, fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    
    for out_dir in out_dirs:
        chart2_path = out_dir / "ablation_safety.png"
        plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Charts generated successfully in docs/images and paper/zjuthesis/figures")

if __name__ == "__main__":
    plot_ablation_results()
