import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 10,
    "axes.linewidth": 1.2,
})

MODEL_COLORS = {
    "qwen3.6-plus": "#62B66B",
    "glm-5v-turbo": "#4C9ACD",
    "gpt-5.4": "#F4A261",
    "gemini-3.1-pro-preview": "#C65D5D",
}

MODEL_MARKERS = {
    "qwen3.6-plus": "^",
    "glm-5v-turbo": "s",
    "gpt-5.4": "D",
    "gemini-3.1-pro-preview": "o",
}


def load_config_data(model_name, config_file):
    config_file = Path(config_file)

    if not config_file.exists():
        raise FileNotFoundError(f"Missing file: {config_file}")

    df = pd.read_csv(config_file)
    df.columns = df.columns.str.strip().str.lower()
    df["model_name"] = model_name

    required_cols = {
        "prompt_name",
        "accuracy",
        "recall_defect",
        "recall_perfect",
        "avg_latency_sec",
        "label_reason_consistency_rate_auto",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{model_name} data file missing columns: {missing}")

    return df


def draw_single_metric(model_dfs, prompts, metric, ylabel, save_name, ylim=None):
    fig, ax = plt.subplots(figsize=(6, 4))

    for model_name, df in model_dfs.items():
        df_ordered = df.set_index("prompt_name").reindex(prompts)

        color = MODEL_COLORS.get(model_name, None)
        marker = MODEL_MARKERS.get(model_name, "o")

        ax.plot(
            prompts,
            df_ordered[metric],
            label=model_name,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=7,
        )

    ax.set_xlabel("Prompt Setting")
    ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(save_name, bbox_inches="tight")
    plt.close(fig)


def save_shared_legend(save_name="shared_legend.png", ncol=4):
    model_order = [
        "qwen3.6-plus",
        "glm-5v-turbo",
        "gpt-5.4",
        "gemini-3.1-pro-preview",
    ]

    handles = []
    for model_name in model_order:
        handles.append(
            Line2D(
                [0], [0],
                color=MODEL_COLORS[model_name],
                marker=MODEL_MARKERS[model_name],
                linewidth=2.2,
                markersize=7,
                linestyle="-",
                label=model_name
            )
        )

    fig = plt.figure(figsize=(12, 1.2))
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.legend(
        handles=handles,
        loc="center",
        ncol=ncol,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.6,
        handletextpad=0.6,
        borderaxespad=0.0
    )

    fig.savefig(save_name, bbox_inches="tight", transparent=True, pad_inches=0.05)
    plt.close(fig)


def draw_label_reason_consistency(model_dfs, prompts, save_name):
    fig, ax = plt.subplots(figsize=(6, 4))

    for model_name, df in model_dfs.items():
        df_ordered = df.set_index("prompt_name").reindex(prompts)

        color = MODEL_COLORS.get(model_name, None)
        marker = MODEL_MARKERS.get(model_name, "o")

        ax.plot(
            prompts,
            df_ordered["label_reason_consistency_rate_auto"],
            label=model_name,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=7,
            linestyle="-",
        )

    ax.set_xlabel("Prompt Setting")
    ax.set_ylabel("Label–Reason Consistency Rate")
    ax.set_ylim(-0.05, 1.15)

    ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(save_name, bbox_inches="tight")
    plt.close(fig)


def draw_metric_figures(model_dfs, prompts):
    draw_single_metric(
        model_dfs=model_dfs,
        prompts=prompts,
        metric="accuracy",
        ylabel="Accuracy",
        save_name=BASE / "accuracy.png",
        ylim=(-0.05, 1.15),
    )

    draw_single_metric(
        model_dfs=model_dfs,
        prompts=prompts,
        metric="recall_defect",
        ylabel=r"$\mathrm{Recall}_{\mathrm{defect}}$",
        save_name=BASE / "recall_defect.png",
        ylim=(-0.05, 1.15),
    )

    draw_single_metric(
        model_dfs=model_dfs,
        prompts=prompts,
        metric="recall_perfect",
        ylabel=r"$\mathrm{Recall}_{\mathrm{perfect}}$",
        save_name=BASE / "recall_perfect.png",
        ylim=(-0.05, 1.15),
    )

    draw_single_metric(
        model_dfs=model_dfs,
        prompts=prompts,
        metric="avg_latency_sec",
        ylabel="Average Latency per Image (s)",
        save_name=BASE / "latency.png",
    )

    draw_label_reason_consistency(
        model_dfs=model_dfs,
        prompts=prompts,
        save_name=BASE / "label_reason_consistency.png",
    )


file_paths = {
    "qwen3.6-plus": DATA_DIR / "qwen3.6-plus.csv",
    "glm-5v-turbo": DATA_DIR / "glm-5v-turbo.csv",
    "gpt-5.4": DATA_DIR / "gpt-5.4.csv",
    "gemini-3.1-pro-preview": DATA_DIR / "gemini-3.1-pro-preview.csv",
}

if __name__ == "__main__":
    prompts = ["A", "B", "C-2", "C-4", "C-8"]

    model_dfs = {}
    for model_name, path in file_paths.items():
        model_dfs[model_name] = load_config_data(model_name, path)

    draw_metric_figures(model_dfs, prompts)

    save_shared_legend(BASE / "shared_legend_horizontal.png", ncol=4)

    print("Saved figures:")
    print("accuracy.png")
    print("recall_defect.png")
    print("recall_perfect.png")
    print("latency.png")
    print("label_reason_consistency.png")
    print("shared_legend_horizontal.png")
