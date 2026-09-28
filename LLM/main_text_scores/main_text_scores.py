import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent

SIDE_VIEW_BATCH_FILE = BASE / "side_view_batch.csv"
TOP_VIEW_BATCH_FILE = BASE / "top_view_batch.csv"
SIDE_VIEW_RETRY_FILE = BASE / "side_view_retry.csv"

OUTPUT_BATCH_PNG = BASE / "batch_scores.png"
OUTPUT_RETRY_PNG = BASE / "retry_scores.png"

PROMPT_ORDER = ["A", "B", "C-2", "C-4", "C-8"]

MODEL_ORDER = [
    "qwen3.6-plus",
    "glm-5v-turbo",
    "gpt-5.4",
    "gemini-3.1-pro-preview",
]

WEIGHTS = {
    "recall_defect": 0.35,
    "accuracy": 0.20,
    "recall_perfect": 0.20,
    "consistency": 0.20,
    "latency_score": 0.05,
}

CNN_METRICS = {
    "accuracy": 0.95,
    "recall_defect": 1.00,
    "recall_perfect": 0.90,
    "consistency": 1.00,
    "avg_latency_sec": 0.01,
}

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 1.2,
})

PROMPT_COLORS = {
    "A": "#C65D5D",
    "B": "#4C9ACD",
    "C-2": "#62B66B",
    "C-4": "#F4A261",
    "C-8": "#B08CC2",
}

CNN_LINE_COLOR = "#7F7F7F"


def load_score_file(csv_path):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    required_cols = {"model_name", "prompt_name", "avg_latency_sec"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{path.name} is missing required columns: {required_cols - set(df.columns)}")

    if "weighted_score_100" in df.columns:
        df["score"] = pd.to_numeric(df["weighted_score_100"], errors="coerce")
    elif "weighted_score" in df.columns:
        df["score"] = pd.to_numeric(df["weighted_score"], errors="coerce") * 100
    else:
        raise ValueError(
            f"{path.name} has neither a 'weighted_score_100' nor a 'weighted_score' column."
        )

    df["avg_latency_sec"] = pd.to_numeric(df["avg_latency_sec"], errors="coerce")
    df["prompt_name"] = df["prompt_name"].astype(str)
    df["model_name"] = df["model_name"].astype(str)

    df_grouped = (
        df.groupby(["model_name", "prompt_name"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_latency=("avg_latency_sec", "mean"),
        )
    )

    return df_grouped


def make_pivot(df_grouped, model_order=None, prompt_order=None):
    if model_order is None:
        model_order = sorted(df_grouped["model_name"].unique().tolist())
    if prompt_order is None:
        prompt_order = sorted(df_grouped["prompt_name"].unique().tolist())

    pivot = df_grouped.pivot_table(
        index="model_name",
        columns="prompt_name",
        values="mean_score",
        aggfunc="mean"
    )

    existing_models = [m for m in model_order if m in pivot.index]
    existing_prompts = [p for p in prompt_order if p in pivot.columns]

    pivot = pivot.reindex(index=existing_models, columns=existing_prompts)

    return pivot


def compute_cnn_score(df_grouped):
    lat_series = pd.to_numeric(df_grouped["mean_latency"], errors="coerce").dropna()

    if len(lat_series) == 0:
        latency_score = 1.0
    else:
        lat_min = lat_series.min()
        lat_max = lat_series.max()

        if abs(lat_max - lat_min) < 1e-12:
            latency_score = 1.0
        else:
            latency_score = (lat_max - CNN_METRICS["avg_latency_sec"]) / (lat_max - lat_min)
            latency_score = max(0.0, min(1.0, latency_score))

    cnn_score = (
        WEIGHTS["recall_defect"] * CNN_METRICS["recall_defect"]
        + WEIGHTS["accuracy"] * CNN_METRICS["accuracy"]
        + WEIGHTS["recall_perfect"] * CNN_METRICS["recall_perfect"]
        + WEIGHTS["consistency"] * CNN_METRICS["consistency"]
        + WEIGHTS["latency_score"] * latency_score
    ) * 100

    return cnn_score


def draw_grouped_bars(ax, pivot, cnn_score, show_xlabel=False, show_cnn=True):
    models = pivot.index.tolist()
    prompts = pivot.columns.tolist()

    n_models = len(models)
    n_prompts = len(prompts)

    x = np.arange(n_models)
    total_width = 0.82
    bar_width = total_width / max(n_prompts, 1)

    bar_dict = {}

    for i, prompt in enumerate(prompts):
        offset = (i - (n_prompts - 1) / 2) * bar_width
        y = pivot[prompt].values.astype(float)

        bars = ax.bar(
            x + offset,
            y,
            width=bar_width * 0.95,
            label=prompt,
            color=PROMPT_COLORS.get(prompt, None),
            alpha=0.9
        )
        bar_dict[prompt] = bars

    for model_idx, model in enumerate(models):
        row = pivot.loc[model]

        if row.notna().sum() == 0:
            continue

        best_prompt = row.idxmax()
        best_score = row.max()

        best_bar = bar_dict[best_prompt][model_idx]
        best_bar.set_edgecolor("black")
        best_bar.set_linewidth(2.2)

        bar_center = best_bar.get_x() + best_bar.get_width() / 2
        bar_height = best_bar.get_height()

        ax.text(
            bar_center,
            bar_height + 1.0,
            f"{best_score:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="black"
        )

    if show_cnn:
        ax.axhline(
            y=cnn_score,
            color=CNN_LINE_COLOR,
            linestyle="--",
            linewidth=1.8,
            alpha=0.95,
            label="CNN (97.0)"
        )

    ax.set_ylabel("Average Score")
    ax.set_xticks(x)

    if show_xlabel:
        ax.set_xticklabels(models, rotation=0)
        ax.set_xlabel("Model")
    else:
        ax.set_xticklabels([])

    ymax = np.nanmax(pivot.values)

    if show_cnn:
        ymax = max(ymax, cnn_score)
    ax.set_ylim(0, max(100, ymax * 1.18))

    ax.grid(axis="y", linestyle="--", alpha=0.4)


def unique_legend_from_ax(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    return list(unique.values()), list(unique.keys())


def draw_batch_figure():
    side_view_df = load_score_file(SIDE_VIEW_BATCH_FILE)
    top_view_df = load_score_file(TOP_VIEW_BATCH_FILE)

    side_view_pivot = make_pivot(side_view_df, MODEL_ORDER, PROMPT_ORDER)
    top_view_pivot = make_pivot(top_view_df, MODEL_ORDER, PROMPT_ORDER)

    side_view_cnn = compute_cnn_score(side_view_df)
    top_view_cnn = compute_cnn_score(top_view_df)

    print("\n===== SIDE VIEW BATCH =====")
    print(side_view_pivot.round(2).to_string())
    print(f"CNN score = {side_view_cnn:.2f}")

    print("\n===== TOP VIEW BATCH =====")
    print(top_view_pivot.round(2).to_string())
    print(f"CNN score = {top_view_cnn:.2f}")

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(9, 6),
        sharex=True,
        constrained_layout=True
    )

    draw_grouped_bars(axes[0], side_view_pivot, side_view_cnn, show_xlabel=False, show_cnn=True)
    draw_grouped_bars(axes[1], top_view_pivot, top_view_cnn, show_xlabel=True, show_cnn=False)

    handles, labels = unique_legend_from_ax(axes[0])
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 1)
    )

    fig.savefig(OUTPUT_BATCH_PNG, bbox_inches="tight")
    print(f"\n[Saved batch figure] {OUTPUT_BATCH_PNG}")


def draw_retry_figure():
    retry_df = load_score_file(SIDE_VIEW_RETRY_FILE)

    retry_pivot = make_pivot(retry_df, MODEL_ORDER, PROMPT_ORDER)
    retry_cnn = compute_cnn_score(retry_df)

    print("\n===== SIDE VIEW RETRY =====")
    print(retry_pivot.round(2).to_string())
    print(f"CNN score = {retry_cnn:.2f}")

    fig, ax = plt.subplots(
        figsize=(9, 4),
        constrained_layout=True
    )

    draw_grouped_bars(ax, retry_pivot, retry_cnn, show_xlabel=True)

    handles, labels = unique_legend_from_ax(ax)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99)
    )

    fig.savefig(OUTPUT_RETRY_PNG, bbox_inches="tight")
    print(f"\n[Saved retry figure] {OUTPUT_RETRY_PNG}")


def main():
    draw_batch_figure()
    draw_retry_figure()


if __name__ == "__main__":
    main()
