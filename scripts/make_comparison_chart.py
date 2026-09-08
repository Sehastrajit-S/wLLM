"""Generates the two separate benchmark images used in the README:
  - benchmarks/throughput_chart.png  (grouped bar charts only)
  - benchmarks/throughput_table.png  (data table only)

Comparing plain HuggingFace `transformers` (no serving engine at all) vs
wLLM vs real vLLM 0.28.0, all measured on the same physical RTX 3060 (HF and
wLLM natively on Windows, vLLM via WSL2 with torch.compile + CUDA graphs
enabled).

Source data is hardcoded from actual benchmark runs (scripts/benchmark_baseline_hf.py
for HF, scripts/benchmark_cuda_graph.py for wLLM, scripts/vllm_bench_wsl.py
run inside WSL2 for vLLM) -- see the conversation/commit history for the raw
run logs this was transcribed from.

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
HF = {
    "Qwen2.5-0.5B": [41.5, 161.0, 673.2],
    "Qwen2.5-1.5B": [26.9, 133.6, 582.5],
    "Qwen2.5-3B": [28.5, 114.7, 380.1],
}
WLLM = {
    "Qwen2.5-0.5B": [157.8, 565.6, 2096.1],
    "Qwen2.5-1.5B": [72.8, 271.9, 1033.8],
    "Qwen2.5-3B": [41.5, 145.0, 533.5],
}
VLLM = {
    "Qwen2.5-0.5B": [247.9, 905.3, 3144.1],
    "Qwen2.5-1.5B": [90.3, 371.5, 1405.8],
    "Qwen2.5-3B": [48.8, 180.9, 673.8],
}

COLOR_HF = "#94a3b8"
COLOR_WLLM = "#2563eb"
COLOR_VLLM = "#dc2626"


def make_chart(out_path: str) -> None:
    fig = plt.figure(figsize=(14, 6.5))
    gs = fig.add_gridspec(1, 3, wspace=0.18, top=0.78, bottom=0.12)
    bar_width = 0.26
    x = np.arange(len(BATCH_SIZES))

    for i, model in enumerate(MODELS):
        ax = fig.add_subplot(gs[0, i])
        hf_vals = HF[model]
        wllm_vals = WLLM[model]
        vllm_vals = VLLM[model]

        ax.bar(x - bar_width, hf_vals, bar_width, label="Plain HF transformers (no serving engine)", color=COLOR_HF)
        ax.bar(x, wllm_vals, bar_width, label="wLLM (CUDA graphs)", color=COLOR_WLLM)
        ax.bar(x + bar_width, vllm_vals, bar_width, label="vLLM 0.28.0 (WSL2, compiled)", color=COLOR_VLLM)

        for xi, v in zip(x - bar_width, hf_vals):
            ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="#475569")
        for xi, v in zip(x, wllm_vals):
            ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=COLOR_WLLM, fontweight="bold")
        for xi, v in zip(x + bar_width, vllm_vals):
            ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=COLOR_VLLM)

        ax.set_title(model, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"batch={b}" for b in BATCH_SIZES])
        ax.set_ylabel("tokens/sec" if i == 0 else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    fig.text(0.5, 0.97, "Decode throughput: no serving engine vs wLLM vs vLLM", ha="center", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.935, "HF and wLLM: native Windows. vLLM: WSL2, torch.compile + CUDA graphs. Higher is better.", ha="center", fontsize=9.5, color="#475569")
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.9), ncol=3, fontsize=11, frameon=False)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {out_path}")


def make_table(out_path: str) -> None:
    rows = []
    for model in MODELS:
        for j, b in enumerate(BATCH_SIZES):
            hf_v, wllm_v, vllm_v = HF[model][j], WLLM[model][j], VLLM[model][j]
            wllm_ratio = wllm_v / hf_v
            vllm_ratio = vllm_v / hf_v
            rows.append([
                model, str(b), f"{hf_v:.1f}", f"{wllm_v:.1f}", f"{wllm_ratio:.2f}x",
                f"{vllm_v:.1f}", f"{vllm_ratio:.2f}x",
            ])

    col_labels = [
        "Model", "Batch", "Plain HF\n(tok/s)", "wLLM\n(tok/s)", "wLLM vs\nHF",
        "vLLM\n(tok/s)", "vLLM vs\nHF",
    ]

    fig, ax_table = plt.subplots(figsize=(11, 4.3))
    ax_table.axis("off")

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
                cell.set_text_props(color=COLOR_WLLM, fontweight="bold")
            if c == 6:
                cell.set_text_props(color=COLOR_VLLM)

    fig.text(0.5, 0.04, "Data: scripts/benchmark_baseline_hf.py (HF), scripts/benchmark_cuda_graph.py (wLLM), scripts/vllm_bench_wsl.py (vLLM, run inside WSL2)", ha="center", fontsize=8, color="#94a3b8", style="italic")

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {out_path}")


def main() -> None:
    make_chart("benchmarks/throughput_chart.png")
    make_table("benchmarks/throughput_table.png")


if __name__ == "__main__":
    main()
