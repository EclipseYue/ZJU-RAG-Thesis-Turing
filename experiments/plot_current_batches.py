import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
BATCH_ROOT = ROOT / "data" / "results" / "batches"
FIG_ROOT = ROOT / "paper" / "zjuthesis" / "figures"


def load_matrix(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data[0]
    if isinstance(data, dict) and "matrix" in data:
        return data["matrix"][0]
    return data


def plot_route_a_quality():
    files = [
        ("Heuristic-20", BATCH_ROOT / "2026-04-28-route-a-server" / "route_a_hotpotqa_heuristic_smoke_matrix.json"),
        ("RealAPI-20", BATCH_ROOT / "2026-04-28-route-a-server" / "route_a_hotpotqa_realapi_smoke_latest_matrix.json"),
        ("RealAPI-100", BATCH_ROOT / "2026-04-28-route-a-server" / "route_a_hotpotqa_realapi_100_matrix.json"),
        ("RealAPI-300", ROOT / "data" / "results" / "route_a_hotpotqa_realapi_300.json"),
    ]
    labels, ems, f1s, recalls, hits = [], [], [], [], []
    for label, path in files:
        row = load_matrix(path)
        labels.append(label)
        ems.append(row["ExactMatch"])
        f1s.append(row["F1_Score"])
        recalls.append(row["SupportRecall@K"])
        hits.append(row["SupportAllHit@K"])

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    x = range(len(labels))
    width = 0.36
    axes[0].bar([i - width / 2 for i in x], ems, width=width, label="EM", color="#4C78A8")
    axes[0].bar([i + width / 2 for i in x], f1s, width=width, label="F1", color="#F58518")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Route A Answer Quality")
    axes[0].legend()
    for i, (em, f1) in enumerate(zip(ems, f1s)):
        axes[0].text(i - width / 2, em + 0.6, f"{em:.1f}", ha="center", fontsize=9)
        axes[0].text(i + width / 2, f1 + 0.6, f"{f1:.1f}", ha="center", fontsize=9)

    axes[1].bar([i - width / 2 for i in x], recalls, width=width, label="SupportRecall@K", color="#54A24B")
    axes[1].bar([i + width / 2 for i in x], hits, width=width, label="SupportAllHit@K", color="#E45756")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Percentage")
    axes[1].set_title("Route A Retrieval Stability")
    axes[1].legend()
    for i, (r, h) in enumerate(zip(recalls, hits)):
        axes[1].text(i - width / 2, r + 0.6, f"{r:.1f}", ha="center", fontsize=9)
        axes[1].text(i + width / 2, h + 0.6, f"{h:.1f}", ha="center", fontsize=9)

    fig.tight_layout()
    target = FIG_ROOT / "route_a_quality_comparison.png"
    plt.savefig(target, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_current_batch_comparison():
    files = [
        ("RouteA\nHeuristic-20", BATCH_ROOT / "2026-04-28-route-a-server" / "route_a_hotpotqa_heuristic_smoke_matrix.json"),
        ("RouteA\nRealAPI-20", BATCH_ROOT / "2026-04-28-route-a-server" / "route_a_hotpotqa_realapi_smoke_latest_matrix.json"),
        ("RouteA\nRealAPI-100", BATCH_ROOT / "2026-04-28-route-a-server" / "route_a_hotpotqa_realapi_100_matrix.json"),
        ("RouteA\nRealAPI-300", ROOT / "data" / "results" / "route_a_hotpotqa_realapi_300.json"),
        ("HybridQA\nTable-50", ROOT / "data" / "results" / "route_a_hybridqa_text_table_smoke_50.json"),
        ("Legacy A\nSmoke-20", BATCH_ROOT / "2026-04-28-legacy-server-smoke" / "legacy_a_baseline_smoke_matrix.json"),
        ("Legacy A3\nCoVe-20", BATCH_ROOT / "2026-04-28-legacy-server-smoke" / "legacy_a3_cove_smoke_matrix.json"),
    ]
    labels, f1s, ems = [], [], []
    for label, path in files:
        row = load_matrix(path)
        labels.append(label)
        ems.append(row["ExactMatch"])
        f1s.append(row["F1_Score"])

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11.5, 5.2))
    x = range(len(labels))
    width = 0.36
    plt.bar([i - width / 2 for i in x], ems, width=width, label="EM", color="#72B7B2")
    plt.bar([i + width / 2 for i in x], f1s, width=width, label="F1", color="#B279A2")
    plt.xticks(list(x), labels)
    plt.ylabel("Score")
    plt.title("Current Server Batch Comparison")
    plt.legend()
    for i, (em, f1) in enumerate(zip(ems, f1s)):
        plt.text(i - width / 2, em + 0.6, f"{em:.1f}", ha="center", fontsize=9)
        plt.text(i + width / 2, f1 + 0.6, f"{f1:.1f}", ha="center", fontsize=9)
    plt.tight_layout()
    target = FIG_ROOT / "current_server_batch_comparison.png"
    plt.savefig(target, dpi=300, bbox_inches="tight")
    plt.close()
    return target


def plot_latency_noanswer():
    files = [
        ("RouteA\nHeuristic-20", BATCH_ROOT / "2026-04-28-route-a-server" / "route_a_hotpotqa_heuristic_smoke_matrix.json"),
        ("RouteA\nRealAPI-20", BATCH_ROOT / "2026-04-28-route-a-server" / "route_a_hotpotqa_realapi_smoke_latest_matrix.json"),
        ("RouteA\nRealAPI-100", BATCH_ROOT / "2026-04-28-route-a-server" / "route_a_hotpotqa_realapi_100_matrix.json"),
        ("RouteA\nRealAPI-300", ROOT / "data" / "results" / "route_a_hotpotqa_realapi_300.json"),
        ("HybridQA\nTable-50", ROOT / "data" / "results" / "route_a_hybridqa_text_table_smoke_50.json"),
        ("Legacy A\nSmoke-20", BATCH_ROOT / "2026-04-28-legacy-server-smoke" / "legacy_a_baseline_smoke_matrix.json"),
        ("Legacy A3\nCoVe-20", BATCH_ROOT / "2026-04-28-legacy-server-smoke" / "legacy_a3_cove_smoke_matrix.json"),
    ]
    labels, latencies, no_answers = [], [], []
    for label, path in files:
        row = load_matrix(path)
        labels.append(label)
        latencies.append(row.get("Avg_Latency_ms", 0.0))
        no_answers.append(row.get("No_Answer_Rate_Percent", 0.0))

    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(11.5, 5.2))
    x = range(len(labels))
    bars = ax1.bar(list(x), latencies, color="#4C78A8", alpha=0.75, label="Avg Latency (ms)")
    ax1.set_ylabel("Avg Latency (ms)", color="#4C78A8")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.tick_params(axis="y", labelcolor="#4C78A8")
    for i, bar in enumerate(bars):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 25, f"{latencies[i]:.0f}", ha="center", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(list(x), no_answers, color="#E45756", marker="o", linewidth=2.2, label="No-Answer Rate (%)")
    ax2.set_ylabel("No-Answer Rate (%)", color="#E45756")
    ax2.tick_params(axis="y", labelcolor="#E45756")
    for i, value in enumerate(no_answers):
        ax2.text(i, value + 2.0, f"{value:.1f}%", color="#E45756", ha="center", fontsize=9)

    plt.title("Latency and Rejection Behavior Across Current Batches")
    fig.tight_layout()
    target = FIG_ROOT / "current_batch_latency_noanswer.png"
    plt.savefig(target, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return target


if __name__ == "__main__":
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    targets = [
        plot_route_a_quality(),
        plot_current_batch_comparison(),
        plot_latency_noanswer(),
    ]
    for target in targets:
        print(f"Saved: {target}")
