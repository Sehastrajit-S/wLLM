"""Generates benchmarks/wllm_vs_vllm.png: a table + grouped bar charts
comparing wLLM (before/after the vLLM-style PagedAttention kernel rewrite)
against real vLLM 0.28.0, all measured on the same physical RTX 3060 (wLLM
natively on Windows, vLLM via WSL2 with torch.compile + CUDA graphs enabled).

Source data is hardcoded from actual benchmark runs (scripts/benchmark_cuda_graph.py
for wLLM, scripts/vllm_bench_wsl.py run inside WSL2 for vLLM) -- see the
conversation/commit history for the raw run logs this was transcribed from.

Run: python scripts/make_comparison_chart.py
"""
import sys

sys.path.insert(0, "src")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B"]
BATCH_SIZES = [1, 4, 16]

# tokens/sec, [model][batch_size]
WLLM_OLD = {
    "Qwen2.5-0.5B": [115.1, 432.8, 1473.2],
    "Qwen2.5-1.5B": [53.9, 205.8, 789.3],
    "Qwen2.5-3B": [33.4, 110.1, 379.7],
}
WLLM_NEW = {
    "Qwen2.5-0.5B": [157.8, 565.6, 2096.1],
    "Qwen2.5-1.5B": [72.8, 271.9, 1033.8],
    "Qwen2.5-3B": [41.5, 145.0, 533.5],
}
VLLM = {
    "Qwen2.5-0.5B": [247.9, 905.3, 3144.1],
    "Qwen2.5-1.5B": [90.3, 371.5, 1405.8],
    "Qwen2.5-3B": [48.8, 180.9, 673.8],
}

COLOR_OLD = "#94a3b8"
COLOR_NEW = "#2563eb"
COLOR_VLLM = "#dc2626"


def main() -> None:
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.1, 1], hspace=0.35, top=0.85, bottom=0.06)

    # --- Bar charts: one subplot per model, batch size on x-axis ---
    gs_top = gs[0].subgridspec(1, 3, wspace=0.18)
    bar_width = 0.26
    x = np.arange(len(BATCH_SIZES))

    for i, model in enumerate(MODELS):
        ax = fig.add_subplot(gs_top[0, i])
        old_vals = WLLM_OLD[model]
        new_vals = WLLM_NEW[model]
        vllm_vals = VLLM[model]

        ax.bar(x - bar_width, old_vals, bar_width, label="wLLM (original kernel)", color=COLOR_OLD)
        ax.bar(x, new_vals, bar_width, label="wLLM (vLLM-style kernel)", color=COLOR_NEW)
        ax.bar(x + bar_width, vllm_vals, bar_width, label="vLLM 0.28.0 (WSL2, compiled)", color=COLOR_VLLM)

        for xi, v in zip(x - bar_width, old_vals):
            ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="#475569")
        for xi, v in zip(x, new_vals):
            ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=COLOR_NEW, fontweight="bold")
        for xi, v in zip(x + bar_width, vllm_vals):
            ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=COLOR_VLLM)

        ax.set_title(model, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"batch={b}" for b in BATCH_SIZES])
        ax.set_ylabel("tokens/sec" if i == 0 else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    fig.text(0.5, 0.975, "wLLM vs vLLM: PagedAttention decode throughput, same RTX 3060", ha="center", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.955, "wLLM: native Windows. vLLM: WSL2, torch.compile + CUDA graphs. Higher is better.", ha="center", fontsize=9.5, color="#475569")
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, fontsize=11, frameon=False)

    # --- Table ---
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off")

    rows = []
    for model in MODELS:
        for j, b in enumerate(BATCH_SIZES):
            old_v, new_v, vllm_v = WLLM_OLD[model][j], WLLM_NEW[model][j], VLLM[model][j]
            speedup = new_v / old_v
            vllm_ratio = vllm_v / new_v
            rows.append([
                model, str(b), f"{old_v:.1f}", f"{new_v:.1f}", f"{speedup:.2f}x",
                f"{vllm_v:.1f}", f"{vllm_ratio:.2f}x",
            ])

    col_labels = [
        "Model", "Batch", "wLLM old\n(tok/s)", "wLLM new\n(tok/s)", "Kernel\nspeedup",
        "vLLM\n(tok/s)", "vLLM still\nahead by",
    ]
    table = ax_table.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.9)

    for c in range(len(col_labels)):
        cell = table[0, c]
        cell.set_facecolor("#1e293b")
        cell.set_text_props(color="white", fontweight="bold")

    for r in range(1, len(rows) + 1):
        for c in range(len(col_labels)):
            cell = table[r, c]
            cell.set_facecolor("#f8fafc" if r % 2 == 0 else "white")
            if c == 4:
                cell.set_text_props(color=COLOR_NEW, fontweight="bold")
            if c == 6:
                cell.set_text_props(color=COLOR_VLLM)

    fig.text(0.5, 0.025, "Data: scripts/benchmark_cuda_graph.py (wLLM) and scripts/vllm_bench_wsl.py (vLLM, run inside WSL2)", ha="center", fontsize=8, color="#94a3b8", style="italic")

    out_path = "benchmarks/wllm_vs_vllm.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
