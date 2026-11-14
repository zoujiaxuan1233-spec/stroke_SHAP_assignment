"""Task 1: run the supplied teaching script unchanged, then extend its EDA.

Usage: python code/00_data_exploration.py --data data/healthcare-dataset-stroke-data.csv
The source script is retained under original/xai-series-master.
"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import io
import json
import runpy
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]


def run_eda(data_path, output_dir, source_dir=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(source_dir or ROOT / "original/xai-series-master")
    # A runtime adapter supplies the CSV location and saves plt.show() output.
    # No lines of the retained teaching script or DataLoader are rewritten.
    sys.path.insert(0, str(source_dir))
    import utils
    original_load = utils.DataLoader.load_dataset
    original_show = plt.show
    figures = []

    def adapted_load(self, path=None):
        return original_load(self, str(data_path))

    def save_show(*args, **kwargs):
        index = len(figures)
        name = f"original_{index + 1:02d}_{pd.read_csv(data_path).columns[index]}.png"
        plt.gcf().set_size_inches(7, 4)
        plt.tight_layout()
        plt.savefig(output_dir / name, dpi=150, bbox_inches="tight")
        figures.append(name)
        plt.close("all")

    log = io.StringIO()
    try:
        utils.DataLoader.load_dataset = adapted_load
        plt.show = save_show
        with contextlib.redirect_stdout(log):
            namespace = runpy.run_path(str(source_dir / "00_data_exploration.py"), run_name="__main__")
    finally:
        utils.DataLoader.load_dataset = original_load
        plt.show = original_show
        sys.path.remove(str(source_dir))
    (output_dir / "original_execution.log").write_text(log.getvalue(), encoding="utf-8")
    namespace["data_loader"].data.head().to_csv(output_dir / "original_preprocessed_head.csv", index=False)
    d = pd.read_csv(data_path)
    d.describe(include="all").to_csv(output_dir / "descriptive_statistics.csv")
    d.isna().sum().rename("missing_count").to_csv(output_dir / "missing_values.csv")
    for col in CATEGORICAL:
        d[col].value_counts(dropna=False).rename_axis(col).rename("count").to_csv(output_dir / f"counts_{col}.csv")
    d.groupby("stroke")[NUMERIC].agg(["count", "mean", "std", "median", "min", "max"]).to_csv(output_dir / "numeric_by_stroke.csv")
    info = {
        "shape": list(d.shape), "sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "missing": {k: int(v) for k, v in d.isna().sum().items()},
        "class_counts": {str(k): int(v) for k, v in d.stroke.value_counts().items()},
        "prevalence": float(d.stroke.mean()), "duplicate_rows": int(d.duplicated().sum()),
        "duplicate_ids": int(d.id.duplicated().sum()),
        "duplicate_predictor_target_rows": int(d.drop(columns="id").duplicated().sum()),
        "unknown_smoking": int(d.smoking_status.eq("Unknown").sum()),
        "categories": {c: d[c].value_counts().to_dict() for c in CATEGORICAL},
        "numeric_summary": d[NUMERIC].describe().to_dict(),
        "group_medians": d.groupby("stroke")[NUMERIC].median().to_dict(),
        "bmi_missing_by_stroke": d.assign(missing=d.bmi.isna()).groupby("stroke").missing.agg(["sum", "mean"]).to_dict(),
        "original_script_executed": True, "original_figure_count": len(figures),
    }
    (output_dir / "eda.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    counts = d.stroke.value_counts().sort_index()
    axs[0, 0].bar(["No stroke", "Recorded stroke"], counts, color=["#386a8b", "#c44959"])
    axs[0, 0].set(title="Recorded stroke status", ylabel="Number of records")
    for i, value in enumerate(counts):
        axs[0, 0].text(i, value + 60, f"{value:,} ({100*value/len(d):.2f}%)", ha="center")
    for ax, col, label in zip([axs[0, 1], axs[0, 2], axs[1, 0]], NUMERIC,
                              ["Age (years)", "Average glucose (recorded units)", "BMI (recorded units)"]):
        ax.hist(d.loc[d.stroke.eq(0), col].dropna(), bins=25, density=True, alpha=.55, label="No stroke", color="#386a8b")
        ax.hist(d.loc[d.stroke.eq(1), col].dropna(), bins=25, density=True, alpha=.55, label="Recorded stroke", color="#c44959")
        ax.set(xlabel=label, ylabel="Density", title=label.split(" (")[0] + " by stroke status")
        ax.legend(fontsize=9)
    axs[1, 1].bar(["BMI missing", "Smoking unknown"], [d.bmi.isna().mean()*100, d.smoking_status.eq("Unknown").mean()*100], color="#386a8b")
    axs[1, 1].set(title="Unavailable information", ylabel="Percentage of all records (%)", ylim=(0, 35))
    for i, value in enumerate([d.bmi.isna().mean()*100, d.smoking_status.eq("Unknown").mean()*100]):
        axs[1, 1].text(i, value+.6, f"{value:.2f}%", ha="center")
    rates = d.groupby("hypertension").stroke.agg(["mean", "count"])
    axs[1, 2].bar(["Absent", "Present"], 100*rates["mean"], color="#386a8b")
    axs[1, 2].set(title="Recorded stroke and hypertension", xlabel="Hypertension", ylabel="Recorded stroke prevalence (%)", ylim=(0, 18))
    for i, (_, row) in enumerate(rates.iterrows()):
        axs[1, 2].text(i, row["mean"]*100+.3, f"{100*row['mean']:.1f}%\nn={int(row['count']):,}", ha="center", fontsize=10)
    fig.suptitle("Exploratory analysis of 5,110 supplied records", fontsize=16)
    fig.savefig(output_dir / "eda_panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return info


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data/healthcare-dataset-stroke-data.csv")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/eda")
    a = p.parse_args()
    print(json.dumps(run_eda(a.data, a.output), indent=2))
