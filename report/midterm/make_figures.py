"""Figures for the 10-703 mid-point report.

Sources:
  * Experiment 1, closed-loop counts: official-demonstration reproduction on the GPU
    workstation (Claude Code report, 2026-11-12), 150 initial states per condition.
  * Experiment 1, lambda sweep and calibration: container run, recomputed from stored
    checkpoints and branch labels (data/lambda_sweep.json, data/calib.npz).
  * Sim-to-sim pilot: Phase A scan and Phase C paired differences as reported by Claude Code
    on 2026-11-13 (5 seeds x 150 initial states; bootstrap CIs over seeds and initial states).
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent

for path in Path("/usr/share/fonts/truetype/freefont").glob("FreeSerif*.ttf"):
    font_manager.fontManager.addfont(str(path))

mpl.rcParams.update({
    "font.family": ["FreeSerif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#8a8986",
    "axes.linewidth": 0.6,
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

INK, MUTED = "#1f1f1e", "#52514e"
C = {  # fixed slot per entity
    "pi0": "#2a78d6", "bc": "#eb6834", "error": "#1baf7a", "nohist": "#eda100",
    "smooth": "#e87ba4", "lam": "#6250d6", "expert_perturbed": "#008300", "zero": "#8a8986",
}
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * p, 100 * (centre - half), 100 * (centre + half)


# --------------------------------------------------------------------------- #
# Figure 2: experiment 1 closed-loop success by injected delay
# --------------------------------------------------------------------------- #
def fig_success():
    counts = {  # successes out of 150 at r = 0, 2, 4 (official-demonstration run)
        "pi0": ("$\\pi_0$ (initial BC)", [26, 35, 74]),
        "bc": ("BC fine-tune ($\\lambda=0$)", [85, 89, 127]),
        "smooth": ("BC + smoothness", [83, 99, 127]),
        "error": ("BC + $E$ (selected $\\lambda=10^{-3}$)", [60, 74, 110]),
        "nohist": ("BC + $E$, no history", [58, 91, 119]),
        "lam": ("BC + $E$, fixed $\\lambda=0.03$", [8, 9, 20]),
    }
    fig, ax = plt.subplots(figsize=(6.5, 2.35))
    delays = [0, 2, 4]
    width = 0.13
    keys = list(counts)
    for i, key in enumerate(keys):
        label, ks = counts[key]
        xs = np.arange(3) + (i - (len(keys) - 1) / 2) * width
        stats = [wilson(k, 150) for k in ks]
        mid = [s[0] for s in stats]
        lo = [s[0] - s[1] for s in stats]
        hi = [s[2] - s[0] for s in stats]
        ax.errorbar(xs, mid, yerr=[lo, hi], fmt="o", ms=4.2, color=C[key], ecolor=C[key],
                    elinewidth=1.0, capsize=0, label=label, mec="white", mew=0.6)
    ax.set_xticks(range(3), [f"injected delay $r={d}$" for d in delays])
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, 2.5)
    ax.legend(ncols=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.30),
              handletextpad=0.3, columnspacing=1.2)
    fig.savefig(OUT / "exp1_success.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3: surrogate exploitation (a) and calibration (b), container run
# --------------------------------------------------------------------------- #
def fig_surrogate():
    sweep = {float(k): v for k, v in json.loads((HERE / "data" / "lambda_sweep.json").read_text()).items()}
    weights = sorted(sweep)
    labels = ["0\n(BC)"] + [f"{w:g}" for w in weights[1:]]
    true = np.array([sweep[w]["true"] for w in weights]) * 1e6
    pred = np.array([sweep[w]["pred"] for w in weights]) * 1e6

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.55), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(weights))
    a.plot(x, true, "-o", color=C["bc"], ms=4, lw=1.6, mec="white", mew=0.6)
    a.plot(x, pred, "-o", color=C["error"], ms=4, lw=1.6, mec="white", mew=0.6)
    a.text(5.6, 150, "true $J$ (simulator)", color=C["bc"], ha="right", va="bottom", fontsize=8)
    a.text(0.8, 21, "predicted $E$", color=C["error"], ha="right", va="top", fontsize=8)
    a.set_yscale("log")
    a.set_xticks(x, labels)
    a.set_xlabel("penalty weight $\\lambda$")
    a.set_ylabel("mean cost of policy chunk ($\\mathrm{mm}^2$)")
    a.set_title("(a) Optimising $E$ raises the true deviation", loc="left")
    a.grid(axis="y", which="minor", color="#f0efeb", lw=0.4)

    d = np.load(HERE / "data" / "calib.npz", allow_pickle=True)
    pred_mm = np.sqrt(np.maximum(d["pred"], 0)) * 1e3
    true_mm = np.sqrt(d["true"]) * 1e3
    rng = np.random.default_rng(0)
    order = [("policy_perturbed", "perturbed policy", "#b4b2a9"), ("expert_perturbed", "perturbed expert", "#8fc9a6"),
             ("policy", "policy $\\pi_0$", C["pi0"]), ("expert", "expert (true $J=0$)", C["bc"])]
    for key, label, colour in order:
        m = d["typ"] == key
        idx = np.nonzero(m)[0]
        if len(idx) > 300:
            idx = rng.choice(idx, 300, replace=False)
        jitter = rng.uniform(-0.25, 0.25, len(idx)) if key == "expert" else 0
        b.scatter(true_mm[idx] + jitter, pred_mm[idx], s=5 if key.endswith("perturbed") else 7, color=colour,
                  alpha=0.75, lw=0, label=label, zorder=3 if not key.endswith("perturbed") else 2)
    lim = 25
    b.plot([0, lim], [0, lim], color=MUTED, lw=0.7, ls="--", zorder=1)
    b.set_xlim(-0.6, lim)
    b.set_ylim(-0.6, lim)
    b.set_xlabel("true $\\sqrt{J}$ (mm)")
    b.set_ylabel("predicted $\\sqrt{E}$ (mm)")
    b.set_title("(b) Held-out calibration of $E$", loc="left")
    b.grid(axis="both")
    b.legend(frameon=False, loc="upper left", handletextpad=0.1, markerscale=1.8, borderaxespad=0.2)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "exp1_surrogate.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 4: sim-to-sim pilot
# --------------------------------------------------------------------------- #
def fig_sim2sim():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.7), gridspec_kw={"width_ratios": [1.2, 1]})

    scan = [
        ("nominal", "nominal", 85, "zero"),
        ("0.8", "actuation", 80, "pi0"), ("0.6", "actuation", 74, "pi0"),
        ("0.4", "actuation", 51, "pi0"), ("0.3", "actuation", 24, "pi0"),
        ("1", "latency", 30, "bc"), ("2", "latency", 20, "bc"),
        ("0.5/2", "contact", 85, "error"), ("0.3/3", "contact", 74, "error"), ("0.2/4", "contact", 72, "error"),
    ]
    positions, ticks, x = [], [], 0.0
    previous = None
    for label, family, k, colour in scan:
        if previous is not None and family != previous:
            x += 0.7
        mid, lo, hi = wilson(k, 150)
        chosen = (family, label) in {("actuation", "0.4"), ("latency", "1")}
        a.errorbar([x], [mid], yerr=[[mid - lo], [hi - mid]], fmt="D" if chosen else "o", ms=5 if chosen else 4,
                   color=C[colour], ecolor=C[colour], elinewidth=1.0, capsize=0,
                   mec=INK if chosen else "white", mew=0.8 if chosen else 0.6)
        positions.append(x)
        ticks.append(label)
        previous = family
        x += 1.25 if family == "contact" else 1
    gate = 100 * 85 / 150 - 15
    a.axhline(gate, color=MUTED, lw=0.7, ls="--")
    a.text(positions[-1] + 0.3, gate - 2.0, "gate 1: $-15$ pp vs. nominal", color=MUTED, fontsize=7.5, ha="right", va="top")
    a.set_xticks(positions, ticks, fontsize=7.2)
    for family, centre in (("stiffness\nscale $s$", np.mean(positions[1:5])), ("delay\n$d$ steps", np.mean(positions[5:7])),
                           ("friction/mass", np.mean(positions[7:10]))):
        a.text(centre, -0.17, family, ha="center", va="top", fontsize=8, color=MUTED, transform=a.get_xaxis_transform())
    a.set_ylim(0, 75)
    a.set_ylabel("target success (%)")
    a.set_title("(a) Phase A: which shifts create a gap", loc="left")

    rows = [
        ("Gap penalty, demo states", {"latency": (-0.4, -6.7, 6.0), "actuation": (-7.7, -13.5, -2.3)}),
        ("Gap penalty, logged states", {"latency": (-15.1, -21.3, -9.2)}),
        ("Imitation, ckpt on target", {"actuation": (3.2, -0.9, 8.7)}),
        ("Same-budget fine-tune", {"latency": (6.1, -0.5, 13.2), "actuation": (9.1, 0.3, 17.7)}),
        ("Smoothness penalty", {"latency": (8.9, 3.7, 14.7), "actuation": (5.2, 0.1, 10.8)}),
    ]
    shift_style = {"latency": ("delay $d=1$", C["bc"], "o", -0.14), "actuation": ("stiffness $s=0.4$", C["pi0"], "s", 0.14)}
    for i, (name, values) in enumerate(rows):
        y = len(rows) - 1 - i
        for shift, (mid, lo, hi) in values.items():
            label, colour, marker, offset = shift_style[shift]
            b.errorbar([mid], [y + offset], xerr=[[mid - lo], [hi - mid]], fmt=marker, ms=4.2, color=colour,
                       ecolor=colour, elinewidth=1.1, capsize=0, mec="white", mew=0.6,
                       label=label if i == 0 else None)
    b.axvline(0, color=MUTED, lw=0.7)
    b.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=7.8)
    b.set_xlabel("target success vs. zero-shot (pp, 95% CI)")
    b.grid(axis="x")
    b.grid(axis="y", visible=False)
    b.set_xlim(-24, 20)
    b.set_title("(b) Phase C: transfer to the target", loc="left")
    b.legend(frameon=False, loc="lower left", handletextpad=0.2, borderaxespad=0.1)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "sim2sim.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_success()
    fig_surrogate()
    fig_sim2sim()
    print("done")
