import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results"
FIGURES = ROOT / "paper" / "zjuthesis" / "figures"


def find_feedback_result() -> Path | None:
    preferred = [
        RESULTS / "verification_feedback_study_hotpotqa_50_v2.json",
        RESULTS / "verification_feedback_study_hotpotqa_50.json",
        RESULTS / "verification_feedback_study_hotpotqa.json",
    ]
    for path in preferred:
        if path.exists():
            return path
    candidates = sorted(
        RESULTS.glob("verification_feedback_study_hotpotqa*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_matrix_rows() -> list[dict]:
    rows = []
    candidates = [
        ("RouteA RealAPI-100", RESULTS / "batches" / "2026-04-28-route-a-server" / "route_a_hotpotqa_realapi_100_matrix.json"),
        ("Legacy A", RESULTS / "batches" / "2026-04-28-legacy-server-smoke" / "legacy_a_baseline_smoke_matrix.json"),
        ("Legacy A3 CoVe", RESULTS / "batches" / "2026-04-28-legacy-server-smoke" / "legacy_a3_cove_smoke_matrix.json"),
    ]
    feedback_path = find_feedback_result()
    for label, path in candidates:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        row = data[0] if isinstance(data, list) else data["matrix"][0]
        rows.append({"label": label, **row})
    if feedback_path.exists():
        data = json.loads(feedback_path.read_text(encoding="utf-8"))
        for name, payload in data.get("variants", {}).items():
            rows.append({"label": name, **payload["summary"]})
    return rows


def plot_tradeoffs(rows: list[dict]) -> list[Path]:
    sns.set_theme(style="whitegrid")
    targets = []
    if not rows:
        return targets

    plt.figure(figsize=(7.5, 5.2))
    for row in rows:
        plt.scatter(row.get("No_Answer_Rate_Percent", 0.0), row.get("F1_Score", 0.0), s=90)
        plt.text(row.get("No_Answer_Rate_Percent", 0.0) + 0.8, row.get("F1_Score", 0.0), row["label"], fontsize=9)
    plt.xlabel("No-Answer Rate (%)")
    plt.ylabel("F1 Score")
    plt.title("F1 vs. Rejection Tradeoff")
    plt.tight_layout()
    target = FIGURES / "tradeoff_f1_rejection.png"
    plt.savefig(target, dpi=300, bbox_inches="tight")
    plt.close()
    targets.append(target)

    plt.figure(figsize=(7.5, 5.2))
    for row in rows:
        latency = row.get("Avg_Latency_ms", 0.0)
        plt.scatter(latency, row.get("F1_Score", 0.0), s=90)
        plt.text(latency + 35, row.get("F1_Score", 0.0), row["label"], fontsize=9)
    plt.xlabel("Average Latency (ms)")
    plt.ylabel("F1 Score")
    plt.title("F1 vs. Latency Tradeoff")
    plt.tight_layout()
    target = FIGURES / "tradeoff_f1_latency.png"
    plt.savefig(target, dpi=300, bbox_inches="tight")
    plt.close()
    targets.append(target)
    return targets


def plot_calibration() -> Path | None:
    path = find_feedback_result()
    if not path:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for name, payload in data.get("variants", {}).items():
        for record in payload.get("records", []):
            records.append({
                "variant": name,
                "confidence": float(record.get("avg_confidence", 0.0)),
                "correct": 1.0 if record.get("f1", 0.0) >= 0.5 else 0.0,
            })
    if not records:
        return None

    bins = [i / 10 for i in range(11)]
    points = []
    for idx in range(len(bins) - 1):
        lo, hi = bins[idx], bins[idx + 1]
        bucket = [r for r in records if lo <= r["confidence"] < hi or (idx == 9 and r["confidence"] == hi)]
        if bucket:
            points.append(((lo + hi) / 2, sum(r["correct"] for r in bucket) / len(bucket), len(bucket)))

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 5.2))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    plt.plot([x for x, _, _ in points], [y for _, y, _ in points], marker="o", label="Observed")
    for x, y, n in points:
        plt.text(x, y + 0.03, f"n={n}", ha="center", fontsize=8)
    plt.xlabel("Verification Confidence")
    plt.ylabel("Empirical Correctness (F1>=0.5)")
    plt.title("Verifier Calibration")
    plt.legend()
    plt.tight_layout()
    target = FIGURES / "verifier_calibration.png"
    plt.savefig(target, dpi=300, bbox_inches="tight")
    plt.close()
    return target


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    targets = plot_tradeoffs(load_matrix_rows())
    calibration = plot_calibration()
    if calibration:
        targets.append(calibration)
    for target in targets:
        print(f"Saved: {target}")
