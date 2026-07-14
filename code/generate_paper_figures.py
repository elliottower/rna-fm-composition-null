"""Generate figures for Paper C (E1 druggability gate results).

Figure 1: Gate schematic (TikZ, compiled separately)
Figure 2: Results overview — heatmap of gate pass/fail + untrained-weight bar chart
"""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "e1_druggability", "1000perm")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "protocol", "figures")

MODEL_ORDER = ["RNA-FM", "RiNALMo", "ERNIE-RNA", "SpliceBERT", "UTR-LM", "NTv2", "DNABERT-2"]
MODEL_KEY_MAP = {
    "rnafm": "RNA-FM", "rinalmo": "RiNALMo", "ernierna": "ERNIE-RNA",
    "splicebert": "SpliceBERT", "utrlm": "UTR-LM", "nt": "NTv2", "dnabert2": "DNABERT-2",
}

TARGET_ORDER = [
    "POS-01", "POS-02", "POS-03", "POS-04", "POS-05", "POS-06", "POS-07", "POS-08",
    "NEG-01", "NEG-02", "NEG-03", "NEG-04", "NEG-05",
    "CTL-01", "CTL-02", "CTL-03",
]

TARGET_NAMES = {
    "POS-01": "SMN2", "POS-02": "FMN", "POS-03": "HCV IRES",
    "POS-04": "HIV TAR", "POS-05": "PreQ1", "POS-06": "MALAT1",
    "POS-07": "EV71 IRES", "POS-08": "SARS-CoV-2",
    "NEG-01": "HOTAIR", "NEG-02": "SRA", "NEG-03": "Xist",
    "NEG-04": "Shuffled", "NEG-05": "Random",
    "CTL-01": "HTT CAG17", "CTL-02": "HTT CAG36", "CTL-03": "HTT CAG60",
}


def load_results():
    models = {}
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "e1_*_2026*.json"))):
        if "untrained" in f:
            continue
        d = json.load(open(f))
        key = MODEL_KEY_MAP.get(d["model"], d["model"])
        models[key] = d
    return models


def build_gate_matrices(models):
    n_models = len(MODEL_ORDER)
    n_targets = len(TARGET_ORDER)

    gate_b = np.full((n_models, n_targets), np.nan)
    gate_c = np.full((n_models, n_targets), np.nan)
    gate_c_rho = np.full((n_models, n_targets), np.nan)

    for i, m in enumerate(MODEL_ORDER):
        if m not in models:
            continue
        d = models[m]
        for j, t in enumerate(TARGET_ORDER):
            ac = d["attention_contact"]["per_target"].get(t)
            if ac:
                gate_c[i, j] = 1.0 if ac["gate_c_pass"] else 0.0
                gate_c_rho[i, j] = ac["best_corr"]
            if not d["gate"].get("skipped"):
                gb = d["gate"]["per_target"].get(t)
                if gb:
                    gate_b[i, j] = 1.0 if gb["gate_pass"] else 0.0

    return gate_b, gate_c, gate_c_rho


def figure2_heatmap(models):
    gate_b, gate_c, gate_c_rho = build_gate_matrices(models)

    fig = plt.figure(figsize=(10, 7.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.6], width_ratios=[1, 1],
                           hspace=0.45, wspace=0.35)

    cmap_pf = ListedColormap(["#E53935", "#43A047"])
    cmap_skip = ListedColormap(["#BDBDBD"])

    target_labels = [f"{t}\n{TARGET_NAMES[t]}" for t in TARGET_ORDER]

    # Panel A: Gate B heatmap
    ax_b = fig.add_subplot(gs[0, 0])
    mask_b = np.isnan(gate_b)
    display_b = np.where(mask_b, 0, gate_b)
    im_b = ax_b.imshow(display_b, cmap=cmap_pf, aspect="auto", vmin=0, vmax=1, interpolation="nearest")
    for i in range(gate_b.shape[0]):
        for j in range(gate_b.shape[1]):
            if mask_b[i, j]:
                ax_b.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                               fill=True, facecolor="#E0E0E0", edgecolor="white", linewidth=0.5))
                ax_b.text(j, i, "—", ha="center", va="center", fontsize=7, color="#757575")
            else:
                label = "P" if gate_b[i, j] == 1 else "F"
                color = "white"
                ax_b.text(j, i, label, ha="center", va="center", fontsize=6,
                         fontweight="bold", color=color)

    ax_b.set_xticks(range(len(TARGET_ORDER)))
    ax_b.set_xticklabels([TARGET_NAMES[t] for t in TARGET_ORDER], rotation=45, ha="right", fontsize=6)
    ax_b.set_yticks(range(len(MODEL_ORDER)))
    ax_b.set_yticklabels(MODEL_ORDER, fontsize=7)
    ax_b.set_title("A. Gate B: Composition-null sensitivity", fontsize=9, fontweight="bold", pad=8)

    # Category separators
    ax_b.axvline(7.5, color="black", linewidth=1.5)
    ax_b.axvline(12.5, color="black", linewidth=1.5)

    # Panel B: Gate C heatmap
    ax_c = fig.add_subplot(gs[0, 1])
    im_c = ax_c.imshow(gate_c, cmap=cmap_pf, aspect="auto", vmin=0, vmax=1, interpolation="nearest")
    for i in range(gate_c.shape[0]):
        for j in range(gate_c.shape[1]):
            rho = gate_c_rho[i, j]
            if not np.isnan(rho):
                label = f".{int(rho*100):02d}" if rho < 1 else "1.0"
                color = "white"
                ax_c.text(j, i, label, ha="center", va="center", fontsize=5.5, color=color)

    ax_c.set_xticks(range(len(TARGET_ORDER)))
    ax_c.set_xticklabels([TARGET_NAMES[t] for t in TARGET_ORDER], rotation=45, ha="right", fontsize=6)
    ax_c.set_yticks(range(len(MODEL_ORDER)))
    ax_c.set_yticklabels(MODEL_ORDER, fontsize=7)
    ax_c.set_title("B. Gate C: Attention-contact $\\rho_S$", fontsize=9, fontweight="bold", pad=8)

    ax_c.axvline(7.5, color="black", linewidth=1.5)
    ax_c.axvline(12.5, color="black", linewidth=1.5)

    # Panel C: Untrained-weight comparison (bar chart)
    ax_u = fig.add_subplot(gs[1, 0])
    models_u = ["NTv2", "DNABERT-2"]
    trained = [0.274, 0.262]
    untrained = [0.276, 0.233]
    untrained_err = [0.008, 0.004]
    x = np.arange(len(models_u))
    w = 0.3
    ax_u.bar(x - w/2, trained, w, color="#1565C0", label="Trained", zorder=3)
    ax_u.bar(x + w/2, untrained, w, color="#E0E0E0", edgecolor="#757575",
             yerr=untrained_err, capsize=4, label="Untrained (5-seed)", zorder=3)
    ax_u.set_xticks(x)
    ax_u.set_xticklabels(models_u, fontsize=8)
    ax_u.set_ylabel("Mean $\\rho_S$", fontsize=8)
    ax_u.set_title("C. Untrained-weight control", fontsize=9, fontweight="bold", pad=8)
    ax_u.legend(fontsize=7, loc="upper right")
    ax_u.set_ylim(0, 0.40)
    ax_u.grid(axis="y", alpha=0.3, zorder=0)
    ax_u.annotate("$\\Delta = -0.002$\nCI $[-0.011, +0.007]$",
                  xy=(0, 0.30), fontsize=6.5, ha="center", va="bottom", color="#424242")
    ax_u.annotate("$\\Delta = +0.029$\nCI $[+0.024, +0.034]$",
                  xy=(1, 0.30), fontsize=6.5, ha="center", va="bottom", color="#424242")
    ax_u.set_ylim(0, 0.40)

    # Panel D: Probing accuracy bar chart
    ax_p = fig.add_subplot(gs[1, 1])
    probing_ba = {
        "RNA-FM": 0.625, "RiNALMo": 0.783, "ERNIE-RNA": 0.781,
        "SpliceBERT": 0.743, "UTR-LM": 0.720, "NTv2": 0.576, "DNABERT-2": 0.574,
    }
    bars = [probing_ba[m] for m in MODEL_ORDER]
    colors = ["#1565C0"] * 5 + ["#FF8F00"] * 2
    x_p = np.arange(len(MODEL_ORDER))
    ax_p.bar(x_p, bars, color=colors, zorder=3)
    ax_p.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="Chance", zorder=2)
    ax_p.set_xticks(x_p)
    ax_p.set_xticklabels(MODEL_ORDER, rotation=30, ha="right", fontsize=7)
    ax_p.set_ylabel("Balanced accuracy", fontsize=8)
    ax_p.set_title("D. Probing accuracy (stem vs. loop)", fontsize=9, fontweight="bold", pad=8)
    ax_p.set_ylim(0.4, 0.85)
    ax_p.grid(axis="y", alpha=0.3, zorder=0)
    ax_p.legend(fontsize=7)

    # Legend for heatmaps
    legend_elements = [
        Patch(facecolor="#43A047", label="Pass"),
        Patch(facecolor="#E53935", label="Fail"),
        Patch(facecolor="#E0E0E0", edgecolor="#757575", label="Skipped (BPE)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3,
              fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.98))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "figure2_results_overview.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {out_path}")

    out_png = os.path.join(OUTPUT_DIR, "figure2_results_overview.png")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    print(f"Saved: {out_png}")

    plt.close(fig)


if __name__ == "__main__":
    models = load_results()
    print(f"Loaded {len(models)} models: {list(models.keys())}")
    figure2_heatmap(models)
