"""Aggregate the per-seed outputs of several full runs into one summary and figures.

Input: a directory holding files named ``seed<N>_<file>`` as downloaded from the Modal volume
(``results.md``, ``eval_episodes_{main,flow,sel}.parquet``, ``eval_true_J_*.parquet``,
``eval_boundaries_*.parquet``, ``selection_audit.json``, ``flow_gradient_audit*.json``,
``gradient_audit_summary.json``, ``pi0_baseline.json``, ``pi0_flow_baseline.json``,
``error_report_history.json``).  A run directory of a single seed can be used too with
``--run-dir`` (its ``eval_*_<tag>.parquet`` shards are merged).

Every paired comparison is computed *within* a seed on the shared initial states (the same
150 states for every policy), then summarised across seeds with a t-interval (n = number of
seeds).  So a confidence interval here answers "would another training run reproduce this",
which the single-run intervals in ``results.md`` cannot.

Outputs in ``--out``: ``summary.md``, ``aggregate.json`` and the figures
``fig_success_by_delay``, ``fig_paired_success``, ``fig_anchor_J``, ``fig_selection_audit``,
``fig_flow_gradient_audit`` (PDF + PNG).
"""

from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {
    "pi0": "MLP pi0", "bc": "MLP BC fine-tune (lambda = 0)", "error": "MLP + error net, selected lambda",
    "nohist": "MLP + error net (no history)", "smooth": "MLP + smoothness, selected beta",
    "error_w0.03": "MLP + error net, lambda = 0.03",
    "pi0_flow": "Flow pi0", "bc_flow": "Flow fine-tune (lambda = 0)", "error_flow": "Flow + error net, selected lambda",
    "smooth_flow": "Flow + smoothness, selected beta", "error_flow_w0.3": "Flow + error net, lambda = 0.3",
    "select_best8_flow": "Flow + critic selects best of 8", "select_best32_flow": "Flow + critic selects best of 32",
    "select_worst8_flow": "Flow + critic selects worst of 8",
}
ORDER = list(LABELS)
MLP = ["pi0", "bc", "error", "nohist", "smooth", "error_w0.03"]
FLOW = ["pi0_flow", "bc_flow", "error_flow", "smooth_flow", "error_flow_w0.3", "select_best8_flow", "select_best32_flow", "select_worst8_flow"]
COLORS = {
    "pi0": "#2a78d6", "bc": "#eb6834", "error": "#1baf7a", "nohist": "#eda100", "smooth": "#e87ba4", "error_w0.03": "#008300",
    "pi0_flow": "#2a78d6", "bc_flow": "#eb6834", "error_flow": "#1baf7a", "smooth_flow": "#e87ba4", "error_flow_w0.3": "#008300",
    "select_best8_flow": "#6250d6", "select_best32_flow": "#2f2a7a", "select_worst8_flow": "#b5651d",
}
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def reference(name: str) -> str:
    return "bc_flow" if name.endswith("_flow") else "bc"


def t_ci(values) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    n = len(v)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "sd": float("nan")}
    if n == 1:
        return {"mean": float(v[0]), "lo": float("nan"), "hi": float("nan"), "n": 1, "sd": float("nan")}
    half = T95.get(n, 1.96) * v.std(ddof=1) / np.sqrt(n)
    return {"mean": float(v.mean()), "lo": float(v.mean() - half), "hi": float(v.mean() + half), "n": int(n), "sd": float(v.std(ddof=1))}


def fmt(ci: dict, scale: float = 1.0, digits: int = 1, unit: str = "") -> str:
    if ci["n"] == 0:
        return "–"
    if ci["n"] == 1:
        return f"{ci['mean'] * scale:.{digits}f}{unit}"
    return f"{ci['mean'] * scale:+.{digits}f}{unit} [{ci['lo'] * scale:+.{digits}f}, {ci['hi'] * scale:+.{digits}f}]"


def fmt_abs(ci: dict, scale: float = 1.0, digits: int = 1, unit: str = "") -> str:
    if ci["n"] == 0:
        return "–"
    if ci["n"] == 1:
        return f"{ci['mean'] * scale:.{digits}f}{unit}"
    return f"{ci['mean'] * scale:.{digits}f}{unit} [{ci['lo'] * scale:.{digits}f}, {ci['hi'] * scale:.{digits}f}]"


# ----------------------------------------------------------------------------- loading
def read_seed_dir(directory: Path) -> dict[int, dict]:
    seeds: dict[int, dict] = {}
    for path in sorted(directory.iterdir()):
        m = re.match(r"seed(\d+)_(.+)$", path.name)
        if not m:
            continue
        seeds.setdefault(int(m.group(1)), {})[m.group(2)] = path
    return seeds


def load_json(path: Path | None):
    return json.loads(path.read_text()) if path and path.is_file() else None


def concat(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in paths if p and p.is_file()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_seed(files: dict[str, Path]) -> dict:
    tags = ("main", "flow", "sel")
    return {
        "episodes": concat([files.get(f"eval_episodes_{t}.parquet") for t in tags]),
        "true_J": concat([files.get(f"eval_true_J_{t}.parquet") for t in tags]),
        "boundaries": concat([files.get(f"eval_boundaries_{t}.parquet") for t in tags]),
        "selection_audit": load_json(files.get("selection_audit.json")),
        "flow_audit": load_json(files.get("flow_gradient_audit.json")),
        "flow_audit_fine": load_json(files.get("flow_gradient_audit_lr1e-5.json")),
        "gradient_audit": load_json(files.get("gradient_audit_summary.json")),
        "pi0": load_json(files.get("pi0_baseline.json")),
        "pi0_flow": load_json(files.get("pi0_flow_baseline.json")),
        "error_report": load_json(files.get("error_report_history.json")),
    }


def load_run_dir(run_dir: Path) -> dict:
    """A single run directory with sharded ``eval_*_<tag>.parquet`` files."""
    def shards(prefix: str) -> list[Path]:
        return sorted(run_dir.glob(f"{prefix}_*.parquet"))
    files = {name: run_dir / name for name in (
        "selection_audit.json", "flow_gradient_audit.json", "flow_gradient_audit_lr1e-5.json", "gradient_audit_summary.json",
        "pi0_baseline.json", "pi0_flow_baseline.json", "error_report_history.json")}
    data = load_seed(files)
    data["episodes"], data["true_J"], data["boundaries"] = concat(shards("eval_episodes")), concat(shards("eval_true_J")), concat(shards("eval_boundaries"))
    return data


# ----------------------------------------------------------------------------- per-seed statistics
def seed_success(ep: pd.DataFrame) -> pd.DataFrame:
    """Success rate per (policy, delay) and pooled."""
    ep = ep.drop_duplicates(subset=["policy", "delay", "init_index"])
    rows = ep.groupby(["policy", "delay"])["success"].mean().reset_index()
    pooled = ep.groupby("policy")["success"].mean().reset_index().assign(delay="all")
    return pd.concat([rows, pooled], ignore_index=True)


def seed_paired(ep: pd.DataFrame) -> pd.DataFrame:
    """Paired success difference vs the same-class reference on shared states."""
    ep = ep.drop_duplicates(subset=["policy", "delay", "init_index"])
    table = ep.pivot_table(index=["init_index", "delay"], columns="policy", values="success").astype(float)
    rows = []
    for name in table.columns:
        ref = reference(name)
        if name == ref or ref not in table.columns:
            continue
        diff = (table[name] - table[ref]).dropna()
        for delay in sorted(set(diff.index.get_level_values("delay"))) + ["all"]:
            sub = diff if delay == "all" else diff.xs(delay, level="delay")
            rows.append({"policy": name, "delay": delay, "diff_pp": 100 * float(sub.mean()), "n": int(len(sub))})
    return pd.DataFrame(rows)


def seed_anchor_J(tj: pd.DataFrame) -> pd.DataFrame:
    ok = tj[tj["status"] == "ok"] if "status" in tj else tj
    wide = ok.pivot_table(index=["episode_id", "t"], columns="policy", values="J_m2")
    rows = []
    for name in wide.columns:
        ref = reference(name)
        row = {"policy": name, "mean_J": float(wide[name].mean())}
        if name != ref and ref in wide.columns:
            pair = wide[[name, ref]].dropna()
            row["rel_vs_ref"] = float((pair[name] - pair[ref]).mean() / pair[ref].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def seed_jump(bd: pd.DataFrame) -> pd.DataFrame:
    per = bd.groupby(["policy", "delay", "init_index"])["jump_xyz"].mean().reset_index()
    rows = []
    for name in per["policy"].unique():
        ref = reference(name)
        if name == ref or ref not in set(per["policy"]):
            continue
        t = per.pivot_table(index=["init_index", "delay"], columns="policy", values="jump_xyz")[[name, ref]].dropna()
        rows.append({"policy": name, "rel_jump_vs_ref": float((t[name] - t[ref]).mean() / t[ref].mean())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- aggregation
def main() -> int:
    parser = ArgumentParser(description="Aggregate several seeds")
    parser.add_argument("--dir", type=Path, default=None, help="directory with seed<N>_<file> files")
    parser.add_argument("--run-dir", type=Path, action="append", default=[], help="single-seed run directory (repeatable)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seeds: dict[str, dict] = {}
    if args.dir:
        for seed, files in read_seed_dir(args.dir).items():
            seeds[str(seed)] = load_seed(files)
    for run_dir in args.run_dir:
        seeds[run_dir.name] = load_run_dir(run_dir)
    seeds = {k: v for k, v in seeds.items() if len(v["episodes"])}
    if not seeds:
        raise SystemExit("no seed with evaluation episodes found")
    names = sorted(seeds)
    print(f"{len(names)} seeds: {names}")

    succ = {s: seed_success(d["episodes"]) for s, d in seeds.items()}
    paired = {s: seed_paired(d["episodes"]) for s, d in seeds.items()}
    anchor = {s: seed_anchor_J(d["true_J"]) for s, d in seeds.items() if len(d["true_J"])}
    jumps = {s: seed_jump(d["boundaries"]) for s, d in seeds.items() if len(d["boundaries"])}
    policies = [p for p in ORDER if any(p in set(t["policy"]) for t in succ.values())]
    delays = sorted({d for t in succ.values() for d in t["delay"] if d != "all"})

    def gather(tables: dict, policy: str, column: str, **match):
        out = {}
        for s, t in tables.items():
            sub = t[(t["policy"] == policy)]
            for k, v in match.items():
                sub = sub[sub[k] == v]
            if len(sub):
                out[s] = float(sub.iloc[0][column])
        return out

    summary: dict = {"seeds": names, "delays": delays, "success": {}, "paired_success_pp": {}, "anchor_J": {}, "jump": {}}
    md: list[str] = [f"# Cross-seed summary ({len(names)} seeds: {', '.join(names)})\n",
                     "Intervals are 95 % t-intervals across seeds of a per-seed quantity; paired differences are "
                     "computed within each seed on the shared initial states against the same-class reference "
                     "(`bc` for MLP policies, `bc_flow` for flow policies).\n"]

    # 1. success
    md.append("## 1. Closed-loop success (%, mean across seeds [95 % CI])\n")
    md.append("| Policy | " + " | ".join(f"r = {d}" for d in delays) + " | pooled |")
    md.append("|---|" + "---:|" * (len(delays) + 1))
    for p in policies:
        cells = []
        for d in [*delays, "all"]:
            per_seed = gather(succ, p, "success", delay=d)
            ci = t_ci(per_seed.values())
            summary["success"].setdefault(p, {})[str(d)] = {**ci, "per_seed": per_seed}
            cells.append(fmt_abs(ci, 100, 1))
        md.append(f"| {LABELS.get(p, p)} | " + " | ".join(cells) + " |")

    # 2. paired success
    md.append("\n## 2. Paired success difference vs same-class reference (pp, mean across seeds [95 % CI]; seeds with negative / positive difference)\n")
    md.append("| Policy | " + " | ".join(f"r = {d}" for d in delays) + " | pooled |")
    md.append("|---|" + "---:|" * (len(delays) + 1))
    for p in policies:
        if p == reference(p):
            continue
        cells = []
        for d in [*delays, "all"]:
            per_seed = gather(paired, p, "diff_pp", delay=d)
            ci = t_ci(per_seed.values())
            summary["paired_success_pp"].setdefault(p, {})[str(d)] = {**ci, "per_seed": per_seed}
            neg = sum(v < 0 for v in per_seed.values())
            pos = sum(v > 0 for v in per_seed.values())
            cells.append(f"{fmt(ci, 1, 1)} ({neg}−/{pos}+)" if per_seed else "–")
        md.append(f"| {LABELS.get(p, p)} | " + " | ".join(cells) + " |")

    # 3. anchor J and jumps
    md.append("\n## 3. Test-anchor J (simulator-recomputed, m²) and handover jump, relative to the reference\n")
    md.append("| Policy | mean J (×1e-5) | ΔJ vs reference (%) | Δ mean handover jump vs reference (%) |")
    md.append("|---|---:|---:|---:|")
    for p in policies:
        j_abs = t_ci(gather(anchor, p, "mean_J").values())
        rel = gather(anchor, p, "rel_vs_ref") if p != reference(p) else {}
        rel_ci = t_ci(rel.values())
        jump = gather(jumps, p, "rel_jump_vs_ref") if p != reference(p) else {}
        jump_ci = t_ci(jump.values())
        summary["anchor_J"][p] = {"mean_J": j_abs, "rel_vs_ref": {**rel_ci, "per_seed": rel}}
        summary["jump"][p] = {**jump_ci, "per_seed": jump}
        md.append(f"| {LABELS.get(p, p)} | {fmt_abs(j_abs, 1e5, 2)} | {fmt(rel_ci, 100, 1) if rel else '–'} | {fmt(jump_ci, 100, 1) if jump else '–'} |")

    # 4. gates and audits
    md.append("\n## 4. Gates and audits per seed\n")
    md.append("| Seed | critic test Pearson | 05 gradient gate (policy samples, ΔJ %) | G2 fine (lr 1e-5): ΔJ % at 1 / 5 / 25 updates, gate | G2 coarse (lr 3e-4): ΔJ % at 10 updates | pi0 r0 val | pi0_flow r0 val |")
    md.append("|---|---:|---:|---|---:|---:|---:|")
    audit_rows = {}
    for s in names:
        d = seeds[s]
        er = d["error_report"] or {}
        pearson = er.get("test", {}).get("pearson") if isinstance(er.get("test"), dict) else er.get("pearson")
        ga = d["gradient_audit"] or {}
        g05 = ga.get("test", {}).get("policy", {}) if isinstance(ga.get("test"), dict) else {}
        g05_rel = g05.get("relative_change_grad")
        g05_gate = ga.get("direction_gate", {}).get("pass") if isinstance(ga.get("direction_gate"), dict) else ga.get("pass")
        fine = d["flow_audit_fine"] or {}
        coarse = d["flow_audit"] or {}

        def rel_at(audit, n):
            for row in audit.get("runs", {}).get("gradient", []):
                if row["updates"] == n:
                    return 100 * row["relative"]
            return None
        fine_vals = [rel_at(fine, n) for n in (1, 5, 25)]
        fine_gate = fine.get("gate", {}).get("pass")
        coarse10 = rel_at(coarse, 10)
        pi0 = (d["pi0"] or {}).get("val_closed_loop_r0", {})
        pf = (d["pi0_flow"] or {}).get("val_closed_loop_r0", {})
        audit_rows[s] = {"pearson": pearson, "g05_rel": g05_rel, "g05_gate": g05_gate, "fine": fine_vals, "fine_gate": fine_gate, "coarse10": coarse10}
        fine_txt = " / ".join("–" if v is None else f"{v:+.0f}" for v in fine_vals) + f", {'pass' if fine_gate else 'fail'}"
        md.append(f"| {s} | {pearson:.3f} | {100 * g05_rel:+.0f} %, {'pass' if g05_gate else 'fail'} | {fine_txt} | "
                  f"{'–' if coarse10 is None else f'{coarse10:+.0f} %'} | {pi0.get('successes', '–')}/{pi0.get('episodes', '–')} | {pf.get('successes', '–')}/{pf.get('episodes', '–')} |"
                  if pearson is not None and g05_rel is not None else f"| {s} | – | – | {fine_txt} | – | – | – |")
    summary["audits"] = audit_rows

    # 5. selection audit
    md.append("\n## 5. Selection audit on validation anchors (critic ranking among the policy's own 8 samples)\n")
    md.append("| Seed | within-anchor Spearman | J picked / random / rejected / oracle (×1e-5) | picked − random (%) | picked is oracle | mean rank of pick |")
    md.append("|---|---:|---:|---:|---:|---:|")
    sel_rows = {}
    for s in names:
        sa = seeds[s]["selection_audit"]
        if not sa:
            continue
        mj = sa["mean_J_m2"]
        rel = 100 * sa["picked_minus_random_J_m2"]["mean"] / mj["random"]["mean"]
        sel_rows[s] = {"spearman": sa["within_anchor_spearman"]["mean"], "picked_minus_random_pct": rel,
                       "picked_is_oracle": sa["picked_is_oracle"], "mean_rank": sa["mean_picked_rank"],
                       "J": {k: mj[k]["mean"] for k in ("picked", "random", "rejected", "oracle")}}
        md.append(f"| {s} | {sa['within_anchor_spearman']['mean']:.2f} | "
                  f"{mj['picked']['mean'] * 1e5:.2f} / {mj['random']['mean'] * 1e5:.2f} / {mj['rejected']['mean'] * 1e5:.2f} / {mj['oracle']['mean'] * 1e5:.2f} | "
                  f"{rel:+.0f} | {100 * sa['picked_is_oracle']:.0f} % | {sa['mean_picked_rank']:.1f} / {sa['samples_per_anchor']} |")
    if sel_rows:
        md.append(f"| **mean [CI]** | {fmt_abs(t_ci([r['spearman'] for r in sel_rows.values()]), 1, 2)} | | "
                  f"{fmt(t_ci([r['picked_minus_random_pct'] for r in sel_rows.values()]), 1, 0)} | "
                  f"{fmt_abs(t_ci([100 * r['picked_is_oracle'] for r in sel_rows.values()]), 1, 0)} % | "
                  f"{fmt_abs(t_ci([r['mean_rank'] for r in sel_rows.values()]), 1, 1)} |")
    summary["selection_audit"] = sel_rows

    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    (args.out / "aggregate.json").write_text(json.dumps(summary, indent=2, default=float))
    make_figures(args.out, names, policies, delays, summary, seeds)
    print("\n".join(md))
    print(f"\nwritten to {args.out}")
    return 0


# ----------------------------------------------------------------------------- figures
def make_figures(out: Path, names, policies, delays, summary, seeds) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e3e3e3", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})

    def save(fig, name):
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Figure 1: success by delay, two panels (MLP class, flow class)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, group, title in ((axes[0], MLP, "MLP chunk policies"), (axes[1], FLOW, "Flow-matching chunk policies")):
        group = [p for p in group if p in policies]
        for k, p in enumerate(group):
            offs = (k - (len(group) - 1) / 2) * 0.12
            means = [100 * summary["success"][p][str(d)]["mean"] for d in delays]
            los = [100 * summary["success"][p][str(d)]["lo"] for d in delays]
            his = [100 * summary["success"][p][str(d)]["hi"] for d in delays]
            x = np.array(delays, dtype=float) + offs
            style = "--" if p.startswith("select") or p.startswith("pi0") else "-"
            ax.plot(x, means, style, color=COLORS[p], lw=1.6, marker="o", ms=4, label=LABELS[p].replace("MLP ", "").replace("Flow ", ""))
            if len(names) > 1:
                ax.fill_between(x, los, his, color=COLORS[p], alpha=0.12, lw=0)
            for s in names:
                v = summary["success"][p][str(delays[0])]["per_seed"]
            per = np.array([[100 * summary["success"][p][str(d)]["per_seed"].get(s, np.nan) for d in delays] for s in names])
            ax.plot(np.tile(x, (len(names), 1)).T, per.T, "o", color=COLORS[p], ms=2, alpha=0.45, mew=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("injected delay r (steps)")
        ax.set_xticks(delays)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7, loc="lower right" if group is not FLOW else "lower left")
    axes[0].set_ylabel("closed-loop success (%)")
    fig.suptitle(f"PickCube-v1, 150 shared initial states per seed, {len(names)} seeds (band: 95 % CI across seeds; dots: seeds)", fontsize=9, y=1.02)
    save(fig, "fig_success_by_delay")

    # Figure 2: paired success differences (forest)
    rows = [(p, d) for p in policies if p != reference(p) for d in [*delays, "all"]]
    fig, ax = plt.subplots(figsize=(7.2, 0.26 * len(rows) + 1.2))
    ylabels = []
    for i, (p, d) in enumerate(rows):
        ci = summary["paired_success_pp"][p][str(d)]
        y = len(rows) - i
        ax.errorbar(ci["mean"], y, xerr=[[ci["mean"] - ci["lo"]] if ci["n"] > 1 else [0], [ci["hi"] - ci["mean"]] if ci["n"] > 1 else [0]],
                    fmt="o", color=COLORS[p], ms=4, capsize=2, lw=1.2)
        ax.plot(list(ci["per_seed"].values()), [y] * len(ci["per_seed"]), "|", color=COLORS[p], ms=7, alpha=0.6)
        ylabels.append(f"{LABELS[p]}, r = {d}" if d != "all" else f"{LABELS[p]}, pooled")
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_yticks(range(len(rows), 0, -1))
    ax.set_yticklabels(ylabels, fontsize=7.5)
    ax.set_xlabel("paired success difference vs same-class lambda = 0 fine-tune (pp)")
    ax.set_title(f"Mean across {len(names)} seeds with 95 % CI; ticks: individual seeds", fontsize=9)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_paired_success")

    # Figure 3: anchor J relative to reference
    comp = [p for p in policies if p != reference(p) and summary["anchor_J"][p]["rel_vs_ref"]["n"]]
    fig, ax = plt.subplots(figsize=(6.4, 0.3 * len(comp) + 1.0))
    for i, p in enumerate(comp):
        ci = summary["anchor_J"][p]["rel_vs_ref"]
        y = len(comp) - i
        ax.errorbar(100 * ci["mean"], y, xerr=[[100 * (ci["mean"] - ci["lo"])] if ci["n"] > 1 else [0], [100 * (ci["hi"] - ci["mean"])] if ci["n"] > 1 else [0]],
                    fmt="o", color=COLORS[p], ms=4, capsize=2, lw=1.2)
        ax.plot([100 * v for v in ci["per_seed"].values()], [y] * ci["n"], "|", color=COLORS[p], ms=7, alpha=0.6)
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_yticks(range(len(comp), 0, -1))
    ax.set_yticklabels([LABELS[p] for p in comp], fontsize=7.5)
    ax.set_xscale("symlog", linthresh=50)
    ax.set_xlabel("test-anchor J relative to same-class reference (%); symlog axis")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_anchor_J")

    # Figure 4: selection audit
    sel = summary.get("selection_audit", {})
    if sel:
        keys = ["oracle", "picked", "random", "rejected"]
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        for s in names:
            if s in sel:
                ax.plot(range(4), [1e5 * sel[s]["J"][k] for k in keys], "-o", color="#888", ms=3, lw=0.8, alpha=0.6)
        means = [1e5 * np.mean([sel[s]["J"][k] for s in sel]) for k in keys]
        ax.plot(range(4), means, "-o", color="#1baf7a", ms=6, lw=2, label="mean across seeds")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["oracle pick\n(lowest true J)", "critic pick\n(lowest E)", "random pick", "critic reject\n(highest E)"], fontsize=8)
        ax.set_ylabel("true J of the executed tail (×1e-5 m²)")
        ax.set_title("Selection among the policy's own 8 samples, validation anchors", fontsize=9)
        ax.legend(fontsize=8)
        save(fig, "fig_selection_audit")

    # Figure 5: flow gradient audit: predicted E (top) and simulator J (bottom), both lr settings
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.2), sharex="col")
    for col, (key, title) in enumerate((("flow_audit_fine", "lr = 1e-5"), ("flow_audit", "lr = 3e-4"))):
        for s in names:
            audit = seeds[s].get(key)
            if not audit:
                continue
            grad, rand = audit["runs"]["gradient"], audit["runs"]["random"]
            ups = [r["updates"] for r in grad]
            axes[0, col].plot(ups, [r["predicted_normalized_E_train"] for r in grad], "-o", color="#1baf7a", ms=3, lw=1.2, alpha=0.8,
                              label="critic-gradient steps" if s == names[0] else None)
            axes[0, col].plot(ups, [r["predicted_normalized_E_train"] for r in rand], ":o", color="#888", ms=3, lw=1.0, alpha=0.8,
                              label="random-direction steps" if s == names[0] else None)
            axes[1, col].plot([0, *ups], [1.0, *[1 + r["relative"] for r in grad]], "-o", color="#eb6834", ms=3, lw=1.2, alpha=0.8,
                              label="critic-gradient steps" if s == names[0] else None)
            axes[1, col].plot([0, *ups], [1.0, *[1 + r["relative"] for r in rand]], ":o", color="#888", ms=3, lw=1.0, alpha=0.8,
                              label="random-direction steps" if s == names[0] else None)
        axes[0, col].set_title(f"Adam on the critic term alone, {title}", fontsize=9)
        axes[0, col].set_yscale("log")
        axes[1, col].set_yscale("log")
        axes[1, col].axhline(1, color="#444", lw=0.6)
        axes[1, col].set_xlabel("updates from pi0_flow")
        axes[0, col].legend(fontsize=7, loc="lower left")
    axes[0, 0].set_ylabel("predicted E (normalised)")
    axes[1, 0].set_ylabel("simulator J relative to pi0_flow")
    fig.suptitle("Critic gradient through the flow sampler (one line per seed)", fontsize=9, y=0.995)
    fig.tight_layout()
    save(fig, "fig_flow_gradient_audit")


if __name__ == "__main__":
    raise SystemExit(main())
