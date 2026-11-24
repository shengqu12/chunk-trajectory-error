"""Aggregate held-out branch results and closed-loop evaluation into ``results.md``.

Paired statistics: differences are taken per test anchor (clustered by episode) or per
initial state, with percentile-bootstrap 95% intervals.  Figures go to ``figures/``.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from chunk_trajectory_error.config import load_config
from chunk_trajectory_error.pipeline import bootstrap_ci, out_dir

LABELS = {
    "pi0": "pi0 (initial BC)",
    "bc": "BC fine-tune (lambda = 0)",
    "error": "BC + error net (history), selected lambda",
    "nohist": "BC + error net (no history), selected lambda",
    "smooth": "BC + action smoothness, selected beta",
    "error_w0.03": "BC + error net (history), fixed lambda = 0.03",
    "error_r1": "BC + error net after 1 refresh round",
    "error_sqrt": "BC + error net, sqrt(J) target (exploratory)",
    "error_fullobs": "BC + error net (history, full obs)",
    "pi0_flow": "Flow: pi0 (initial flow matching)",
    "bc_flow": "Flow: fine-tune (lambda = 0)",
    "error_flow": "Flow + error net (history), selected lambda",
    "smooth_flow": "Flow + action smoothness, selected beta",
    "error_flow_w0.3": "Flow + error net (history), fixed lambda = 0.3",
    "select_best8_flow": "Flow (lambda = 0) + critic selects best of 8 chunks",
    "select_best32_flow": "Flow (lambda = 0) + critic selects best of 32 chunks",
    "select_worst8_flow": "Flow (lambda = 0) + critic selects worst of 8 chunks",
}
ORDER = ["pi0", "bc", "error", "nohist", "smooth", "error_w0.03", "error_r1", "error_sqrt", "error_fullobs",
         "pi0_flow", "bc_flow", "error_flow", "smooth_flow", "error_flow_w0.3",
         "select_best8_flow", "select_best32_flow", "select_worst8_flow"]
# Reference categorical palette, fixed slot per policy (colour follows the entity).
COLORS = {
    "pi0": "#2a78d6", "bc": "#eb6834", "error": "#1baf7a", "nohist": "#eda100", "smooth": "#e87ba4",
    "error_w0.03": "#008300", "error_r1": "#6250d6", "error_sqrt": "#e34948",
    "pi0_flow": "#7fb3e6", "bc_flow": "#f3a884", "error_flow": "#7fd3b4", "smooth_flow": "#f3b9d0", "error_flow_w0.3": "#7fc17f",
    "select_best8_flow": "#2a9d8f", "select_best32_flow": "#1d6f66", "select_worst8_flow": "#b5651d",
}


def reference(name: str) -> str:
    """Paired comparisons are made within a policy class: MLP variants against ``bc``,
    flow-matching variants against ``bc_flow``."""
    return "bc_flow" if name.endswith("_flow") else "bc"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def wilson(successes: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z, p = 1.96, successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def mcnemar_exact(a_only: int, b_only: int) -> float:
    from scipy.stats import binomtest

    return 1.0 if a_only + b_only == 0 else float(binomtest(a_only, a_only + b_only, 0.5).pvalue)


def load(run_dir: Path, stem: str) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in sorted(run_dir.glob(f"{stem}_*.parquet"))]
    return pd.concat(frames, ignore_index=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_bars(path: Path, groups: list[str], series: list[str], values: dict, errors: dict | None, ylabel: str, title: str, fmt: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(7.5, 1.6 + 0.42 * len(series) * len(groups) + 1.2 * len(groups)), 5.4), facecolor=SURFACE)
    style(ax)
    series = [name for name in series if name in COLORS]  # the palette has 8 fixed slots; others are table-only
    width = 0.8 / len(series)
    for index, name in enumerate(series):
        xs = np.arange(len(groups)) + (index - (len(series) - 1) / 2) * width
        heights = [values[name][g] for g in groups]
        ax.bar(xs, heights, width=width * 0.86, color=COLORS[name], label=LABELS[name], linewidth=0)
        if errors:
            lows = [max(0.0, values[name][g] - errors[name][g][0]) for g in groups]
            highs = [max(0.0, errors[name][g][1] - values[name][g]) for g in groups]
            ax.errorbar(xs, heights, yerr=[lows, highs], fmt="none", ecolor=MUTED, elinewidth=1, capsize=2)
        for x, h, g in zip(xs, heights, groups):
            top = errors[name][g][1] if errors else h
            ax.text(x, top, " " + format(h, fmt), ha="center", va="bottom", fontsize=7.5, color=INK, rotation=90)
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.margins(y=0.22)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.1 + 0.045 * ((len(series) + 1) // 2))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    parser = ArgumentParser(description="Summarise held-out branch and closed-loop results")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = out_dir(config)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    episodes, boundaries, costs = load(run_dir, "eval_episodes"), load(run_dir, "eval_boundaries"), load(run_dir, "eval_true_J")
    policies = [name for name in ORDER if name in set(episodes["policy"])]
    delays = sorted(episodes["delay"].unique())
    manifest, replay = read_json(run_dir / "manifest.json"), read_json(run_dir / "replay_audit_summary.json")
    baseline, gradient = read_json(run_dir / "pi0_baseline.json"), read_json(run_dir / "gradient_audit_summary.json")
    lines: list[str] = []
    add = lines.append

    add("# Chunk Trajectory Error: ManiSkill PickCube-v1 results\n")
    runtime = manifest["runtime"]
    add(f"Generated by `scripts/08_report.py`. Host: {runtime['os']}; Python {runtime['python']}; PyTorch {runtime['torch']} "
        f"(CUDA available: {runtime['torch_cuda_is_available']}); ManiSkill {runtime['packages']['mani_skill']}; "
        f"SAPIEN {runtime['packages']['sapien']}; simulation `physx_cpu`, one environment.\n")
    chunk = config["chunk"]
    add(f"Chunk settings: H={chunk['horizon']}, e={chunk['execute_steps']}, L={chunk['history_steps']}, K={chunk['evaluation_steps']}, "
        f"train delay r={chunk['train_delay']}. Seed {config['experiment']['seed']}. One training run per variant (no seed repeats).\n")

    # ---- data ------------------------------------------------------------- #
    add("## 1. Data and replay\n")
    splits = json.loads(Path(config["data"]["split_manifest"]).read_text())["splits"]
    used = sum(len(ids) for ids in splits.values())
    add(f"- Demonstrations: {len(manifest['episodes'])} episodes in the file, the first {used} used (`audit.num_episodes`), "
        f"source `{manifest['data_provenance']['raw_source_type']}`; split by episode into {len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])} (train / val / test).")
    if manifest["data_provenance"].get("note"):
        add(f"- Provenance: {manifest['data_provenance']['note']}")
    add(f"- Replay audit ({replay['anchors_audited']} train anchors, tolerance {replay['tolerance_m'] * 1e3:g} mm): "
        + "; ".join(f"`{mode}` max {info['max_ee_err_m'] * 1e3:.4f} mm ({info['gate']})" for mode, info in replay["modes"].items()) + ".")
    for split in ("train", "val", "test"):
        stats = read_json(run_dir / f"branch_stats_{split}.json")
        if stats:
            parts = ", ".join(f"{k} {v['count']} (median sqrt(J) {v['sqrtJ_mm_median']:.2f} mm)" for k, v in stats["by_type"].items())
            add(f"- Branches, {split}: {stats['anchors']} anchors, {stats['branches']} branches, {stats['aborted']} aborted, "
                f"{stats['clipped_candidates']} candidates clipped at the action box. {parts}.")
    add("")

    # ---- error network ---------------------------------------------------- #
    add("## 2. Error network (held-out branches)\n")
    add("| Variant | Split | MAE (m²) | Constant-mean MAE (m²) | MAE of sqrt(J) (mm) | Pearson | Within-anchor Spearman | Best-candidate match |")
    add("|---|---|---:|---:|---:|---:|---:|---:|")
    for path in sorted(run_dir.glob("error_report_*.json")):
        report, tag = read_json(path), path.stem.replace("error_report_", "")
        for split in ("val", "test"):
            r = report[split]
            add(f"| {tag} | {split} | {r['MAE_m2']:.3e} | {r['constant_baseline_MAE_m2']:.3e} | {r['MAE_sqrtJ_mm']:.2f} | "
                f"{r['pearson']:.3f} | {r['within_anchor_spearman_mean']:.3f} | {r['within_anchor_best_candidate_match']:.2f} |")
    add("")
    add("Mean predicted vs. true J on test, by candidate type (m²): this shows the resolution of each network near J = 0.\n")
    add("| Variant | expert (true 0) predicted | policy true | policy predicted |")
    add("|---|---:|---:|---:|")
    for path in sorted(run_dir.glob("error_report_*.json")):
        by_type = read_json(path)["test"]["by_candidate_type"]
        add(f"| {path.stem.replace('error_report_', '')} | {by_type['expert']['mean_pred_J_m2']:.2e} | {by_type['policy']['mean_true_J_m2']:.2e} | {by_type['policy']['mean_pred_J_m2']:.2e} |")
    add("")

    # ---- gradient --------------------------------------------------------- #
    if gradient:
        add("## 3. Gradient audit\n")
        check = gradient["parameter_gradient_check"]
        add(f"- Parameter gradients from the surrogate alone: policy gradient norm {check['policy_grad_norm']:.3e}, finite "
            f"{check['policy_grads_finite']}, non-zero {check['policy_grads_nonzero']}; error-network parameters without gradient "
            f"{check['error_params_without_grad']}. Gate: {'pass' if check['pass'] else 'FAIL'}.")
        add(f"- Step size: L2 norm over the K x 3 continuous action entries (normalised units), chosen on validation per candidate type from {gradient['step_sizes_tried_on_val']}; gripper channel masked.")
        add("")
        add("| Test candidates | Step | n | Mean J before (m²) | ΔJ gradient step (m²) [95% CI] | ΔJ random step (m²) [95% CI] | Gradient − random (m²) [95% CI] | Relative ΔJ (gradient) | Gradient lowers J | Gate |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for key, row in gradient["test"].items():
            g, rnd, p = row["delta_J_grad_m2"], row["delta_J_rand_m2"], row["paired_grad_minus_rand_m2"]
            add(f"| {key} | {row['step_size']:g} | {row['candidates']} | {row['mean_J0_m2']:.3e} | {g['mean']:.3e} [{g['ci95'][0]:.2e}, {g['ci95'][1]:.2e}] | "
                f"{rnd['mean']:.3e} [{rnd['ci95'][0]:.2e}, {rnd['ci95'][1]:.2e}] | {p['mean']:.3e} [{p['ci95'][0]:.2e}, {p['ci95'][1]:.2e}] | "
                f"{row['relative_change_grad']:+.1%} | {row['fraction_grad_lowers_J']:.0%} | {'pass' if gradient['direction_gate']['by_candidate_type'][key] else 'FAIL'} |")
        add("")
        add("Validation, policy samples only (mean ΔJ of the gradient step / of the random step, m²): "
            + "; ".join(f"step {eta}: {row['delta_J_grad_m2']['mean']:.2e} / {row['delta_J_rand_m2']['mean']:.2e}" for eta, row in gradient["val"]["policy"].items()) + ".")
        add(f"\nDirection gate ({gradient['direction_gate']['rule']}): **{'pass' if gradient['direction_gate']['pass'] else 'FAIL'}**.\n")

    # ---- policies --------------------------------------------------------- #
    add("## 4. Policy fine-tuning\n")
    add(f"All variants start from the same pi0 and use the same data, batches, learning rate and {config['policy_finetune']['steps']} update steps. "
        "The weight is chosen on validation by the simulator-recomputed J of the policy's own chunk.\n")
    flow_baseline = read_json(run_dir / "pi0_flow_baseline.json")
    if flow_baseline:
        add(f"Flow-matching policy class (`*_flow` rows): {flow_baseline['parameters']} parameters, "
            f"{config['models']['policy_flow']['sample_steps']} Euler steps per chunk; pi0_flow best val flow-matching loss "
            f"{flow_baseline['best_val_flow_matching_loss']:.4f}, val MSE of one sampled chunk {flow_baseline['val_sample_mse']:.5f} "
            f"(MLP pi0 val BC loss {baseline['best_val_bc_loss']:.5f}); pi0_flow val closed-loop r = 0: "
            f"{flow_baseline['val_closed_loop_r0']['successes']}/{flow_baseline['val_closed_loop_r0']['episodes']} "
            f"(MLP pi0: {baseline['val_closed_loop_r0']['successes']}/{baseline['val_closed_loop_r0']['episodes']}). "
            "Flow variants start from pi0_flow; their imitation term is the flow-matching loss and the auxiliary term acts on a sampled chunk. "
            "`Val BC loss` for flow rows is the MSE of one sampled chunk.\n")
    add("| Variant | Weights tried (val mean J, m²) | Selected | Val BC loss | Grad-norm ratio auxiliary / imitation (last log) |")
    add("|---|---|---:|---:|---:|")
    for name in policies:
        training = read_json(run_dir / f"policy_training_{name}.json")
        if not training:
            continue
        tried = ", ".join(f"{c['weight']:g}: {c['val_true_J_m2_mean']:.2e}" for c in training["candidates"])
        chosen = next(c for c in training["candidates"] if c["weight"] == training["selected_weight"])
        last = chosen["log"][-1]
        ratio = last["grad_norm_weighted_auxiliary"] / last["grad_norm_imitation"] if last["grad_norm_imitation"] else float("nan")
        add(f"| {LABELS[name]} | {tried} | {training['selected_weight']:g} | {chosen['val_bc_loss']:.5f} | {ratio:.2f} |")
    add("")

    # ---- true J on test anchors ------------------------------------------- #
    add("## 5. Expert deviation on held-out test anchors (r = 0)\n")
    add("J is recomputed in simulation: the policy's chunk tail and the expert's actions both start from the same restored state and run K steps.\n")
    wide = costs.pivot_table(index=["episode_id", "t"], columns="policy", values="J_m2").dropna()
    groups = wide.index.get_level_values("episode_id").to_numpy()
    add("| Policy | Mean J (m²) [95% CI] | Mean sqrt(J) (mm) | Paired ΔJ vs same-class fine-tune, lambda = 0 (m²) [95% CI] | Relative |")
    add("|---|---:|---:|---:|---:|")
    j_values, j_errors = {}, {}
    for name in policies:
        mean, lo, hi = bootstrap_ci(wide[name].to_numpy(), groups)
        j_values[name], j_errors[name] = {"test anchors": mean * 1e6}, {"test anchors": (lo * 1e6, hi * 1e6)}
        root = float(np.sqrt(wide[name]).mean() * 1e3)
        ref = reference(name)
        if name == ref or ref not in wide:
            add(f"| {LABELS[name]} | {mean:.3e} [{lo:.2e}, {hi:.2e}] | {root:.2f} | – | – |")
        else:
            d, dlo, dhi = bootstrap_ci((wide[name] - wide[ref]).to_numpy(), groups)
            add(f"| {LABELS[name]} | {mean:.3e} [{lo:.2e}, {hi:.2e}] | {root:.2f} | {d:.3e} [{dlo:.2e}, {dhi:.2e}] | {d / wide[ref].mean():+.1%} |")
    add(f"\n{len(wide)} anchors from {len(np.unique(groups))} test episodes; intervals resample episodes.\n")
    grouped_bars(fig_dir / "test_true_J.png", ["test anchors"], policies, j_values, j_errors, "mean J (mm²)", "Expert deviation J on held-out anchors (lower is better)", ".1f")
    add("![Expert deviation](figures/test_true_J.png)\n")

    # ---- closed loop ------------------------------------------------------ #
    add("## 6. Closed loop\n")
    n_inits = episodes["init_index"].nunique()
    sources = episodes.drop_duplicates("init_index")["init_source"].value_counts().to_dict()
    add(f"{n_inits} initial states per condition ({sources}); identical states and scheduler for every policy; "
        f"episode limit {config['environment']['max_episode_steps']} steps; success is the environment's own flag. "
        "r is an injected scheduling delay; policies were trained at r = 0 only.\n")
    add("### Success\n")
    add("| Policy | " + " | ".join(f"r = {d}" for d in delays) + " |")
    add("|---|" + "---:|" * len(delays))
    s_values, s_errors = {n: {} for n in policies}, {n: {} for n in policies}
    for name in policies:
        cells = []
        for delay in delays:
            sub = episodes[(episodes["policy"] == name) & (episodes["delay"] == delay)]
            wins, n = int(sub["success"].sum()), len(sub)
            lo, hi = wilson(wins, n)
            s_values[name][f"r = {delay}"], s_errors[name][f"r = {delay}"] = 100 * wins / n, (100 * lo, 100 * hi)
            cells.append(f"{wins}/{n} ({wins / n:.0%}) [{lo:.0%}, {hi:.0%}]")
        add(f"| {LABELS[name]} | " + " | ".join(cells) + " |")
    add("\nBrackets are Wilson 95% intervals.\n")
    grouped_bars(fig_dir / "closed_loop_success.png", [f"r = {d}" for d in delays], policies, s_values, s_errors, "success (%)", "Closed-loop success by injected delay", ".0f")
    add("![Success](figures/closed_loop_success.png)\n")
    unplotted = [LABELS[name] for name in policies if name not in COLORS]
    if unplotted:
        add("Not drawn in the figures (tables only): " + ", ".join(unplotted) + ".\n")
    if "bc" in policies or "bc_flow" in policies:
        add("Paired comparison with the same-class fine-tune at lambda = 0 (same initial states); MLP variants against `bc`, flow variants against `bc_flow`:\n")
        add("| Policy | Delay | Success difference (pp) [95% CI] | Only this policy succeeds | Only reference succeeds | McNemar exact p |")
        add("|---|---:|---:|---:|---:|---:|")
        for name in policies:
            ref = reference(name)
            if name == ref or ref not in policies:
                continue
            for delay in [*delays, "all"]:
                sub = episodes if delay == "all" else episodes[episodes["delay"] == delay]
                table = sub.pivot_table(index=["init_index", "delay"], columns="policy", values="success").astype(float)
                diff = (table[name] - table[ref]).to_numpy()
                mean, lo, hi = bootstrap_ci(diff, table.index.get_level_values("init_index").to_numpy())
                a_only, b_only = int((diff > 0).sum()), int((diff < 0).sum())
                add(f"| {LABELS[name]} | {delay} | {100 * mean:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | {a_only} | {b_only} | {mcnemar_exact(a_only, b_only):.3f} |")
        add("")

    add("### Handover jump at executed boundaries\n")
    add("Jump = ||u_b − u_{b−1}||₂ on the three translation channels of the executed, normalised command (PickCube has no rotation channel); gripper reported separately.\n")
    add("| Policy | Delay | Boundaries | Median jump | P95 jump | Median within-chunk step change | Gripper jump > 0 | Mean EE step (mm) | Mean episode steps |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    jump_values, p95_values = {n: {} for n in policies}, {n: {} for n in policies}
    for name in policies:
        for delay in delays:
            edges = boundaries[(boundaries["policy"] == name) & (boundaries["delay"] == delay)]
            runs = episodes[(episodes["policy"] == name) & (episodes["delay"] == delay)]
            jump_values[name][f"r = {delay}"] = float(edges["jump_xyz"].median())
            p95_values[name][f"r = {delay}"] = float(edges["jump_xyz"].quantile(0.95))
            add(f"| {LABELS[name]} | {delay} | {len(edges)} | {edges['jump_xyz'].median():.4f} | {edges['jump_xyz'].quantile(0.95):.4f} | "
                f"{runs['median_within_chunk_jump_xyz'].median():.4f} | {(edges['jump_gripper'] > 0).mean():.1%} | "
                f"{runs['mean_ee_step_m'].mean() * 1e3:.2f} | {runs['steps'].mean():.1f} |")
    add("")
    grouped_bars(fig_dir / "handover_jump_p95.png", [f"r = {d}" for d in delays], policies, p95_values, None, "P95 jump (normalised action units)", "Handover jump at executed boundaries, 95th percentile", ".3f")
    add("![Handover jump](figures/handover_jump_p95.png)\n")
    if "bc" in policies or "bc_flow" in policies:
        add("Paired per-episode mean jump versus the same-class fine-tune at lambda = 0 (same initial state and delay):\n")
        add("| Policy | Delay | Difference in mean jump [95% CI] | Relative |")
        add("|---|---:|---:|---:|")
        per_episode = boundaries.groupby(["policy", "delay", "init_index"])["jump_xyz"].mean().reset_index()
        for name in policies:
            ref = reference(name)
            if name == ref or ref not in policies:
                continue
            for delay in delays:
                table = per_episode[per_episode["delay"] == delay].pivot_table(index="init_index", columns="policy", values="jump_xyz").dropna(subset=[name, ref])
                mean, lo, hi = bootstrap_ci((table[name] - table[ref]).to_numpy())
                add(f"| {LABELS[name]} | {delay} | {mean:+.4f} [{lo:+.4f}, {hi:+.4f}] | {mean / table[ref].mean():+.1%} |")
        add("")

    # ---- deployment ------------------------------------------------------- #
    add("## 7. Deployment path and latency\n")
    add("Evaluation loads policy checkpoints only (weights plus observation normalisation); no error network, expert history or extra sampling is on the inference path.\n")
    add("| Policy | Device | p50 (ms) | p95 (ms) |")
    add("|---|---|---:|---:|")
    for path in sorted(run_dir.glob("eval_latency_*.json")):
        for name, row in read_json(path).items():
            add(f"| {LABELS.get(name, name)} | {row['device']} | {row['p50_ms']:.3f} | {row['p95_ms']:.3f} |")
    add("\nSingle chunk, batch 1, after 200 warm-up calls. This is measured forward time and is unrelated to the injected delay r.\n")

    # ---- acceptance ------------------------------------------------------- #
    if "bc" in policies or "bc_flow" in policies:
        add("## 8. Acceptance check against the same-class fine-tune (lambda = 0)\n")
        add("Rule from the specification: a closed-loop improvement is claimed only if held-out J and handover jump both improve "
            "and the success evidence supports no loss. \"Improves\" below means the paired 95% interval excludes zero in the favourable direction.\n")
        add("| Policy | Held-out J | Handover jump (delays pooled) | Success (delays pooled) | Conclusion |")
        add("|---|---|---|---|---|")
        per_episode = boundaries.groupby(["policy", "delay", "init_index"])["jump_xyz"].mean().reset_index()
        success_table = episodes.pivot_table(index=["init_index", "delay"], columns="policy", values="success").astype(float)

        def verdict(mean: float, lo: float, hi: float, lower_is_better: bool = True) -> str:
            if hi < 0:
                return "improves" if lower_is_better else "worse"
            if lo > 0:
                return "worse" if lower_is_better else "improves"
            return "no detectable change"

        for name in policies:
            ref = reference(name)
            if name in ("bc", "pi0", "bc_flow", "pi0_flow") or ref not in policies or name.startswith("select_worst"):
                continue
            j = bootstrap_ci((wide[name] - wide[ref]).to_numpy(), groups)
            jumps = per_episode.pivot_table(index=["init_index", "delay"], columns="policy", values="jump_xyz").dropna(subset=[name, ref])
            jump = bootstrap_ci((jumps[name] - jumps[ref]).to_numpy(), jumps.index.get_level_values("init_index").to_numpy())
            succ = bootstrap_ci((success_table[name] - success_table[ref]).to_numpy(), success_table.index.get_level_values("init_index").to_numpy())
            v_j, v_jump, v_s = verdict(*j), verdict(*jump), verdict(*succ, lower_is_better=False)
            if v_j == "improves" and v_jump == "improves" and v_s == "improves":
                conclusion = "closed-loop improvement supported"
            elif v_j == "improves" and v_jump == "improves" and v_s != "worse":
                conclusion = "lower J and smoother handovers; no detectable change in task success"
            elif v_j == "improves" and v_jump != "improves":
                conclusion = "changes trajectory deviation only; no evidence on handover"
            elif v_jump == "improves" and v_s == "worse":
                conclusion = "smoother handover at the cost of task success"
            elif v_j == "worse" or v_s == "worse":
                conclusion = f"worse than {ref} fine-tune"
            else:
                conclusion = "no supported improvement"
            add(f"| {LABELS[name]} | {v_j} ({j[0] / wide[ref].mean():+.1%}) | {v_jump} ({jump[0] / jumps[ref].mean():+.1%}) | "
                f"{v_s} ({100 * succ[0]:+.1f} pp [{100 * succ[1]:+.1f}, {100 * succ[2]:+.1f}]) | {conclusion} |")
        add("")

    # ---- weight sweeps (validation) ---------------------------------------- #
    add("## 9. Validation sweeps for every trained variant\n")
    add("Simulator-recomputed mean J of the policy chunk on validation anchors (m²); BC fine-tune is the lambda = 0 reference.\n")
    add("| Run | Weight: val mean J |")
    add("|---|---|")
    for path in sorted(run_dir.glob("policy_training_*.json")):
        training = read_json(path)
        add(f"| {path.stem.replace('policy_training_', '')} | " + ", ".join(f"{c['weight']:g}: {c['val_true_J_m2_mean']:.2e}" for c in training["candidates"]) + " |")
    add("")

    (run_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("results ->", run_dir / "results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
