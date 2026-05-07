import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results"
FIGURES = ROOT / "paper" / "zjuthesis" / "figures"


def load_route_a_short() -> dict | None:
    path = RESULTS / "route_a_hotpotqa_realapi_50_short_answer.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    row = data[0] if isinstance(data, list) else data["matrix"][0]
    return {
        "label": "RouteA strict-short",
        "F1": row.get("F1_Score", 0.0),
        "EM": row.get("ExactMatch", 0.0),
        "NAR": 0.0,
        "Latency": row.get("Avg_Latency_ms", 0.0),
    }


def load_feedback_short() -> list[dict]:
    path = RESULTS / "verification_feedback_study_hotpotqa_50_short_answer.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for name, payload in data.get("variants", {}).items():
        summary = payload["summary"]
        rows.append({
            "label": name.replace("_short_answer", "").replace("_", " "),
            "F1": summary.get("F1_Score", 0.0),
            "EM": summary.get("ExactMatch", 0.0),
            "NAR": summary.get("No_Answer_Rate_Percent", 0.0),
            "Latency": summary.get("Avg_Latency_ms", 0.0),
        })
    return rows


def load_feedback_reference() -> dict | None:
    path = RESULTS / "verification_feedback_study_hotpotqa_50_v3.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["variants"]["verification_feedback"]["summary"]
    return {
        "label": "feedback concise",
        "F1": summary.get("F1_Score", 0.0),
        "EM": summary.get("ExactMatch", 0.0),
        "NAR": summary.get("No_Answer_Rate_Percent", 0.0),
        "Latency": summary.get("Avg_Latency_ms", 0.0),
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = []
    ref = load_feedback_reference()
    if ref:
        rows.append(ref)
    route_short = load_route_a_short()
    if route_short:
        rows.append(route_short)
    rows.extend(load_feedback_short())
    if not rows:
        return

    sns.set_theme(style="whitegrid")
    labels = [row["label"] for row in rows]
    x = range(len(rows))

    plt.figure(figsize=(8.4, 5.0))
    plt.bar([i - 0.18 for i in x], [row["F1"] for row in rows], width=0.36, label="F1")
    plt.bar([i + 0.18 for i in x], [row["EM"] for row in rows], width=0.36, label="EM")
    plt.xticks(list(x), labels, rotation=18, ha="right")
    plt.ylabel("Score")
    plt.title("Short-answer Constraint Ablation")
    plt.legend()
    plt.tight_layout()
    target = FIGURES / "short_answer_ablation.png"
    plt.savefig(target, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {target}")


if __name__ == "__main__":
    main()
