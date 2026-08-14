from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


PROJECT = Path(r"C:\Users\Mazen\Desktop\AAST\Research\Autonomous research")

SUMMARY_CSV = PROJECT / "outputs" / "milestone_5" / "final_kitti_validation" / "tables" / "comparison_summary_full.csv"

OUTPUT_DIR = PROJECT / "outputs" / "milestone_5" / "benchmark_comparison"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def save_bar_chart(df: pd.DataFrame, column: str, title: str, ylabel: str, filename: str, higher_is_better: bool = True):
    plot_df = df.sort_values(column, ascending=not higher_is_better)

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["detector"], plot_df[column])
    plt.title(title)
    plt.xlabel("Detector")
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    out_path = FIG_DIR / filename
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Saved figure:", out_path.relative_to(PROJECT))


def main():
    print("=" * 100)
    print("STEP 30 - Benchmark and comparison outputs")
    print("=" * 100)

    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing full comparison CSV: {SUMMARY_CSV}")

    df = pd.read_csv(SUMMARY_CSV)

    print("Loaded:", SUMMARY_CSV.relative_to(PROJECT))
    print()
    print(df.to_string(index=False))

    # Ranking columns
    df["rank_mAP50_95"] = df["mAP50_95"].rank(ascending=False, method="min").astype(int)
    df["rank_mAP50"] = df["mAP50"].rank(ascending=False, method="min").astype(int)
    df["rank_speed"] = df["mean_inference_ms"].rank(ascending=True, method="min").astype(int)

    # Simple combined score:
    # Normalize accuracy high = good, latency low = good
    df["norm_mAP50_95"] = df["mAP50_95"] / df["mAP50_95"].max()
    df["norm_speed"] = df["mean_inference_ms"].min() / df["mean_inference_ms"]
    df["combined_score"] = (0.75 * df["norm_mAP50_95"]) + (0.25 * df["norm_speed"])
    df["rank_combined"] = df["combined_score"].rank(ascending=False, method="min").astype(int)

    ranked = df.sort_values(["rank_combined", "rank_mAP50_95", "rank_speed"])

    ranked_csv = TABLE_DIR / "detector_ranking_full.csv"
    ranked_json = TABLE_DIR / "detector_ranking_full.json"

    ranked.to_csv(ranked_csv, index=False)
    ranked_json.write_text(json.dumps(ranked.to_dict(orient="records"), indent=2), encoding="utf-8")

    print()
    print("=" * 100)
    print("Combined ranking")
    print("=" * 100)
    print(ranked[[
        "detector",
        "mAP50",
        "mAP50_95",
        "mean_inference_ms",
        "rank_mAP50_95",
        "rank_mAP50",
        "rank_speed",
        "combined_score",
        "rank_combined",
    ]].to_string(index=False))

    print()
    print("Saved ranking CSV:", ranked_csv.relative_to(PROJECT))
    print("Saved ranking JSON:", ranked_json.relative_to(PROJECT))

    save_bar_chart(
        df,
        column="mAP50_95",
        title="Final KITTI Validation mAP50-95",
        ylabel="mAP50-95",
        filename="map50_95_comparison.png",
        higher_is_better=True,
    )

    save_bar_chart(
        df,
        column="mAP50",
        title="Final KITTI Validation mAP50",
        ylabel="mAP50",
        filename="map50_comparison.png",
        higher_is_better=True,
    )

    save_bar_chart(
        df,
        column="mean_inference_ms",
        title="Mean Inference Time per Image",
        ylabel="Milliseconds / image",
        filename="inference_time_comparison.png",
        higher_is_better=False,
    )

    # Class-level chart source table
    class_columns = [
        "detector",
        "Vehicle_AP50_95",
        "Pedestrian_AP50_95",
        "Cyclist_AP50_95",
    ]
    class_df = df[class_columns].copy()
    class_csv = TABLE_DIR / "class_level_ap50_95_full.csv"
    class_df.to_csv(class_csv, index=False)
    print("Saved class-level CSV:", class_csv.relative_to(PROJECT))

    # Markdown report
    best_accuracy = ranked.sort_values("mAP50_95", ascending=False).iloc[0]
    best_speed = ranked.sort_values("mean_inference_ms", ascending=True).iloc[0]
    best_map50 = ranked.sort_values("mAP50", ascending=False).iloc[0]
    best_combined = ranked.sort_values("combined_score", ascending=False).iloc[0]

    report_lines = []
    report_lines.append("# Milestone 5 Benchmark Comparison")
    report_lines.append("")
    report_lines.append("## Final KITTI validation summary")
    report_lines.append("")
    report_lines.append("| Detector | mAP50 | mAP50-95 | Mean inference ms/image |")
    report_lines.append("|---|---:|---:|---:|")

    for _, row in df.sort_values("mAP50_95", ascending=False).iterrows():
        report_lines.append(
            f"| {row['detector']} | {row['mAP50']:.4f} | {row['mAP50_95']:.4f} | {row['mean_inference_ms']:.2f} |"
        )

    report_lines.append("")
    report_lines.append("## Key findings")
    report_lines.append("")
    report_lines.append(
        f"- Best mAP50-95: **{best_accuracy['detector']}** with **{best_accuracy['mAP50_95']:.4f}**."
    )
    report_lines.append(
        f"- Best mAP50: **{best_map50['detector']}** with **{best_map50['mAP50']:.4f}**."
    )
    report_lines.append(
        f"- Fastest detector: **{best_speed['detector']}** with **{best_speed['mean_inference_ms']:.2f} ms/image**."
    )
    report_lines.append(
        f"- Best combined accuracy-speed score: **{best_combined['detector']}**."
    )
    report_lines.append("")
    report_lines.append("## Interpretation")
    report_lines.append("")
    report_lines.append(
        "YOLO is the strongest overall model because it achieves the highest mAP50-95 while also being the fastest detector."
    )
    report_lines.append(
        "RT-DETR achieves the highest mAP50 by a very small margin, but YOLO performs better under stricter IoU thresholds and has much lower inference latency."
    )
    report_lines.append(
        "Faster R-CNN performs better than RetinaNet in both mAP50 and mAP50-95, but it is also the slowest model in the local benchmark."
    )
    report_lines.append("")
    report_lines.append("## Output files")
    report_lines.append("")
    report_lines.append("- `tables/detector_ranking_full.csv`")
    report_lines.append("- `tables/class_level_ap50_95_full.csv`")
    report_lines.append("- `figures/map50_95_comparison.png`")
    report_lines.append("- `figures/map50_comparison.png`")
    report_lines.append("- `figures/inference_time_comparison.png`")

    report_path = OUTPUT_DIR / "MILESTONE_5_BENCHMARK_COMPARISON.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("Saved markdown report:", report_path.relative_to(PROJECT))

    print()
    print("=" * 100)
    print("STEP 30 COMPLETE ✅")
    print("Benchmark comparison outputs created.")
    print("Next: Step 31, write documentation and DOCX.")
    print("=" * 100)


if __name__ == "__main__":
    main()