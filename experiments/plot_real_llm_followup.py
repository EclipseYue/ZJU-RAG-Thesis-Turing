import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results"
FIGURES = ROOT / "paper" / "zjuthesis" / "figures"


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if "matrix" in data:
        return data["matrix"]
    return []


def load_ablation_rows() -> list[dict]:
    rows: list[dict] = []
    for path in [
        RESULTS / "real_llm_text_ablation_100.json",
        RESULTS / "real_llm_hetero_ablation_100.json",
        RESULTS / "real_llm_full_cove_100.json",
    ]:
        rows.extend(_load_rows(path))
    label_map = {
        "A_Baseline": "A Text",
        "A2_Baseline_Adaptive": "A2 Text+Adaptive",
        "A3_Baseline_CoVe": "A3 Text+CoVe",
        "B_Hetero": "B Hetero",
        "C_Adaptive": "C Hetero+Adaptive",
        "D_CoVe_Full": "D Full",
    }
    return [{**row, "Label": label_map.get(row.get("Config", ""), row.get("Config", ""))} for row in rows]


def load_feedback_rows() -> list[dict]:
    path = RESULTS / "real_llm_verification_feedback_100.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for name, payload in data.get("variants", {}).items():
        summary = payload.get("summary", {})
        rows.append({"Label": name.replace("_", " "), **summary})
    return rows


def plot_real_llm_ablation(rows: list[dict]) -> Path | None:
    if not rows:
        return None
    sns.set_theme(style="whitegrid")
    labels = [row["Label"] for row in rows]
    x = range(len(rows))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(10.5, 5.4))
    ax1.bar([i - width / 2 for i in x], [row["F1_Score"] for row in rows], width=width, label="F1", color="#4C78A8")
    ax1.bar([i + width / 2 for i in x], [row["SupportAllHit@K"] for row in rows], width=width, label="SupportAllHit@K", color="#72B7B2")
    ax1.set_ylabel("Score (%)")
    ax1.set_ylim(0, max(max(row["SupportAllHit@K"] for row in rows), max(row["F1_Score"] for row in rows)) + 12)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=18, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(list(x), [row["No_Answer_Rate_Percent"] for row in rows], color="#E45756", marker="o", linewidth=2, label="No-Answer Rate")
    ax2.set_ylabel("No-Answer Rate (%)")
    ax2.set_ylim(0, max(row["No_Answer_Rate_Percent"] for row in rows) + 15)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    ax1.set_title("Real LLM Ablation Follow-up (HotpotQA, N=100)")
    fig.tight_layout()
    target = FIGURES / "real_llm_ablation_followup.png"
    fig.savefig(target, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_feedback(rows: list[dict]) -> Path | None:
    if not rows:
        return None
    sns.set_theme(style="whitegrid")
    labels = [row["Label"] for row in rows]
    x = range(len(rows))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar([i - width / 2 for i in x], [row["F1_Score"] for row in rows], width=width, label="F1", color="#4C78A8")
    ax.bar([i + width / 2 for i in x], [row["No_Answer_Rate_Percent"] for row in rows], width=width, label="No-Answer Rate", color="#E45756")
    ax.set_ylabel("Percent (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.legend()
    ax.set_title("Real LLM Verification Feedback (HotpotQA, N=100)")
    fig.tight_layout()
    target = FIGURES / "real_llm_feedback_followup.png"
    fig.savefig(target, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return target


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    targets = [
        plot_real_llm_ablation(load_ablation_rows()),
        plot_feedback(load_feedback_rows()),
    ]
    for target in targets:
        if target:
            print(f"Saved: {target}")
