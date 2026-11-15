"""Actuation run report: ``runs/sim2sim_pilot/<shift>/results.md`` (Phase B and Phase C, amendments 11-18).

Phase B: budgets, label distribution, gate 2, the one-feature comparison (action roughness and action
magnitude next to G), the off-distribution check, gate 3 on logged val states (verdict) with the
demonstration-state test next to it (``13_audit_gap_demo.py``, sim-to-sim only).
Phase C (only if gates 2 and 3 pass for the network Phase C uses): per-seed table, pooled paired
differences with two-way cluster bootstrap intervals (helpers of ``16_report_sim2sim.py``), gate 4
against zero-shot and against imitation-selected, gate 5, selected weights, target steps spent on
labels and on selection, and a side-by-side with the latency run (read only).
"""

from __future__ import annotations

import importlib.util
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from chunk_trajectory_error.config import load_config

SCRIPTS = Path(__file__).resolve().parent            # scripts/sim2sim/ (stages 10-17)
PIPELINE_SCRIPTS = SCRIPTS.parent                     # scripts/ (stages 00-08)
ROOT = SCRIPTS.parents[1]


def _script_path(name: str) -> Path:
    return SCRIPTS / name if (SCRIPTS / name).is_file() else PIPELINE_SCRIPTS / name


LABELS = {"zero_shot": "Zero-shot (imitation only)", "imitation_selected": "Imitation-selected (checkpoint chosen in target)",
          "gap_demo": "Gap penalty on demo states (gap-demo)", "gap_logged": "Gap penalty on logged states (gap-logged)",
          "finetune": "Same-budget fine-tune", "smooth": "Action smoothness", "magnitude": "Action magnitude"}
SHORT = {"zero_shot": "zero-shot", "imitation_selected": "imitation-selected", "gap_demo": "gap-demo", "gap_logged": "gap-logged",
         "finetune": "fine-tune", "smooth": "smoothness", "magnitude": "magnitude"}
WEIGHT = {"zero_shot": "-", "imitation_selected": "checkpoint step", "gap_demo": "lambda", "gap_logged": "lambda",
          "finetune": "checkpoint step", "smooth": "beta", "magnitude": "beta"}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name[:-3].lstrip("0123456789_"), _script_path(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R16 = load_script("16_report_sim2sim.py")
two_way_ci, matrix, fmt_ci = R16.two_way_ci, R16.matrix, R16.fmt_ci


def q(d: dict, digits: str = ".2e") -> str:
    return " | ".join(f"{d[k]:{digits}}" for k in ("p10", "median", "p90", "mean"))


def phase_b_section(add, pilot, root, shift_name) -> dict:
    gating = pilot.get("phase_b_gating_network", "sqrtJ")
    collection = {s: json.loads((root / f"collection_{s}.json").read_text()) for s in ("train", "val")}
    report = json.loads((root / "gap_report.json").read_text())
    audit = json.loads((root / "gap_audit_sim2sim_only.json").read_text())
    demo_path = root / "gap_demo_diagnostic_sim2sim_only.json"
    demo = json.loads(demo_path.read_text()) if demo_path.is_file() else None
    identity_path = Path(pilot["output_dir"]) / "identity" / "collection_train.json"
    identity = json.loads(identity_path.read_text()) if identity_path.is_file() else None

    add("## Phase B: target logs, gap labels and G")
    add("")
    add("### Budgets")
    add("")
    add("| Split | Target steps | Episodes (perturbed) | Successes unperturbed / perturbed | Truncated by budget | Label samples (ok / aborted) | Steps without label | Source steps for labels | Target restores |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split, c in collection.items():
        su, sp = c["success_unperturbed"], c["success_perturbed"]
        add(f"| {split} | {c['target_steps_used']:,} / {c['budget_steps']:,} | {c['episodes']} ({c['episodes_perturbed']}) | {su[0]}/{su[1]} / {sp[0]}/{sp[1]} | "
            f"{c['episodes_truncated_by_budget']} | {c['label_samples_ok']:,} / {c['label_samples_aborted']} | {c['steps_without_label']:,} | "
            f"{c['source_steps_for_labels']:,} | {c['target_restores']} |")
    add("")
    add(f"Pending-command buffer: none for this shift (logged shape per step {collection['train']['pending_logged']['shape_per_step']}). "
        "The target is never restored for data; labels restore the source to the logged state and replay the logged commanded actions.")
    add("")
    add("### Label distribution (J_gap, m²)")
    add("")
    add("| Labels | n | p10 | median | p90 | mean | max |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    rows = [("train, all", collection["train"]["J_gap_m2"]), ("train, unperturbed", collection["train"]["J_gap_m2_unperturbed"]),
            ("train, perturbed", collection["train"]["J_gap_m2_perturbed"]), ("val, all", collection["val"]["J_gap_m2"]),
            ("cube, train (m²)", collection["train"]["J_gap_cube_m2"])]
    if identity:
        rows.append(("noise floor: identity target, 2,000 steps (shared with latency)", identity["J_gap_m2"]))
    for name, d in rows:
        add(f"| {name} | {d['n']:,} | {d['p10']:.2e} | {d['median']:.2e} | {d['p90']:.2e} | {d['mean']:.2e} | {d['max']:.2e} |")
    s = collection["train"]["sqrtJ_gap_mm"]
    add("")
    add(f"sqrt(J_gap) on train: median {s['median']:.2f} mm, p90 {s['p90']:.2f} mm.")
    add("")

    add("### Gate 2 (validation, logged val states)")
    add("")
    add(f"Rule: {report['gate2']['rule']}. Split: {report['split']['fit_samples']:,} fit samples, {report['split']['checkpoint_holdout_samples']:,} "
        f"checkpoint-holdout samples (every 10th train episode), {report['split']['val_samples']:,} val samples from {report['split']['val_episodes']} episodes.")
    add("")
    add("| Network | MAE (m²) | Constant-mean MAE (m²) | Pearson | Spearman | Lowest decile true / pred (m²) | Spearman unperturbed / perturbed | Gate 2 |")
    add("|---|---:|---:|---:|---:|---:|---:|---|")
    for target, e in report["networks"].items():
        v = e["val"]
        add(f"| {target}{' (used in Phase C)' if target == gating else ''} | {v['MAE_m2']:.2e} | {v['constant_mean_MAE_m2']:.2e} | {v['pearson']:.3f} | {v['spearman']:.3f} | "
            f"{v['lowest_decile_true_gap']['mean_true_m2']:.2e} / {v['lowest_decile_true_gap']['mean_pred_m2']:.2e} | "
            f"{e['val_unperturbed']['spearman']:.3f} / {e['val_perturbed']['spearman']:.3f} | **{'PASS' if e['gate2']['pass'] else 'FAIL'}** |")
    add("")

    add("### One-feature comparison: G vs action roughness and action magnitude (val)")
    add("")
    add(f"Roughness: {report['roughness']['definition']}. Magnitude: {report['magnitude']['definition']}. The one-feature predictors are linear fits of the label "
        "(J, or sqrt(J) squared back) on the feature, fit on all train labels.")
    add("")
    add("| Predictor of the true gap | MAE (m²) | Pearson | Spearman |")
    add("|---|---:|---:|---:|")
    for target, e in report["networks"].items():
        v = e["val"]
        add(f"| G ({target}) | {v['MAE_m2']:.2e} | {v['pearson']:.3f} | {v['spearman']:.3f} |")
    for feature in ("roughness", "magnitude"):
        for name, v in report[feature]["one_feature_predictors"].items():
            add(f"| {name.replace('_', ' ')} | {v['MAE_m2']:.2e} | {v['pearson']:.3f} | {v['spearman']:.3f} |")
    add("")
    add("| Rank correlation (val) | Roughness | Magnitude |")
    add("|---|---:|---:|")
    add(f"| True gap vs feature (Spearman) | {report['roughness']['spearman_true_gap_vs_roughness_val']:.3f} | {report['magnitude']['spearman_true_gap_vs_magnitude_val']:.3f} |")
    add(f"| True gap vs feature (Pearson) | {report['roughness']['pearson_true_gap_vs_roughness_val']:.3f} | {report['magnitude']['pearson_true_gap_vs_magnitude_val']:.3f} |")
    for target, e in report["networks"].items():
        add(f"| G ({target}) vs feature (Spearman) | {e['roughness']['spearman_prediction_vs_roughness']:.3f} | {e['magnitude']['spearman_prediction_vs_magnitude']:.3f} |")
    add("")
    add("| Feature distribution | p10 | median | p90 | mean |")
    add("|---|---:|---:|---:|---:|")
    for feature in ("roughness", "magnitude"):
        r = report[feature]
        for key, name in ((f"{feature}_logged_val", "logged val commands"), (f"{feature}_demo_actions", "demo actions"),
                          (f"{feature}_pi_bc_chunk_on_demo_states", "pi_bc chunk on demo states")):
            add(f"| {feature}, {name} | {q(r[key])} |")
    add("")

    add("### Off-distribution check (G applied to source demonstration states)")
    add("")
    add("| Distribution (m²) | p10 | median | p90 | mean |")
    add("|---|---:|---:|---:|---:|")
    od = report["off_distribution"]
    add(f"| labels, train | {q(od['label_distribution_train_m2'])} |")
    add(f"| labels, val | {q(od['label_distribution_val_m2'])} |")
    for target, e in report["networks"].items():
        for key, name in (("G_on_logged_val_states_m2", "logged val states, logged actions"), ("G_on_demo_states_demo_actions_m2", "demo states, demo actions"),
                          ("G_on_demo_states_pi_bc_chunk_m2", "demo states, pi_bc chunk")):
            add(f"| G ({target}), {name} | {q(e['off_distribution'][key])} |")
    add(f"| history norm, logged val (mm) | {q(od['history_norm_logged_val_mm'], '.2f')} |")
    add(f"| history norm, demo (mm) | {q(od['history_norm_demo_mm'], '.2f')} |")
    add("")

    add("### Gate 3 (sim-to-sim only: target restores)")
    add("")
    add(f"Logged val states (the verdict, amendment 17): {audit['samples']} unperturbed val samples from {audit['episodes']} episodes, pi_bc's chunk; step size "
        f"chosen on the selection half ({audit['step_size_rule']}), tested once on the other half. Rule: {audit['gate3']['rule']}.")
    add("")
    add("| Network | Step (chosen) | Test samples | ΔJ gradient (m²) [95% CI] | ΔJ random (m²) [95% CI] | Gradient − random (m²) [95% CI] | Relative ΔJ | Gradient lowers J | Gate 3 |")
    add("|---|---:|---:|---|---|---|---:|---:|---|")
    for target, t in audit["test"].items():
        g, r, d = t["delta_J_grad_m2"], t["delta_J_rand_m2"], t["paired_grad_minus_rand_m2"]
        add(f"| {target}{' (used in Phase C)' if target == gating else ''} | {t['step_size']} | {t['samples']} | {fmt_ci(g['mean'], *g['ci95'], sci=True)} | "
            f"{fmt_ci(r['mean'], *r['ci95'], sci=True)} | {fmt_ci(d['mean'], *d['ci95'], sci=True)} | {t['relative_change_grad']:+.1%} | "
            f"{t['fraction_grad_lowers_J']:.0%} | **{'PASS' if audit['gate3']['by_network'][target] else 'FAIL'}** |")
    add("")
    add("Selection half, all step sizes (mean ΔJ gradient, m²): " + "; ".join(
        f"{target}: " + ", ".join(f"{eta} → {v['delta_J_grad_m2']['mean']:+.2e}" for eta, v in sel.items()) for target, sel in audit["selection"].items()) + ".")
    add("")
    rf = audit["restore_fidelity_logged_actions"]
    add(f"Restore fidelity (logged actions replayed in the restored target vs the forward-run label, {rf['samples']} samples): median |difference| "
        f"{rf['median_abs_diff_m2']:.2e} m², p90 {rf['p90_abs_diff_m2']:.2e} m² (median label {rf['median_forward_label_m2']:.2e} m²), Spearman "
        f"{rf['spearman_restored_vs_forward']:.4f}. Skipped: {audit['skipped']}. Target restores: {audit['target_restores']:,}.")
    add("")
    if demo:
        p = demo["prediction_vs_truth"]
        add(f"Demonstration states (reported next to the verdict, not part of it): {demo['samples']} samples from {demo['episodes']} held-out train demonstrations, "
            f"pi_bc's chunk, sqrtJ G; source and target restored to the demonstration state; all step sizes reported, none selected. "
            f"G vs truth: Spearman {p['spearman']:.2f}, Pearson {p['pearson']:.2f}, mean prediction / mean truth {p['mean_pred_over_mean_true']:.2f}; "
            f"true gap median {demo['true_gap_demo_states_m2']['median']:.2e} m² (logged val labels {demo['labels_logged_val_m2']['median']:.2e} m²).")
        add("")
        add("| Step | ΔJ gradient (m²) [95% CI] | ΔJ random (m²) [95% CI] | Gradient − random (m²) [95% CI] | Relative ΔJ | Gradient lowers J | Gate-3 rule met |")
        add("|---:|---|---|---|---:|---:|---|")
        for eta, v in demo["step_test"].items():
            g, r, d = v["delta_J_grad_m2"], v["delta_J_rand_m2"], v["paired_grad_minus_rand_m2"]
            met = g["ci95"][1] < 0 and d["ci95"][1] < 0
            add(f"| {eta} | {fmt_ci(g['mean'], *g['ci95'], sci=True)} | {fmt_ci(r['mean'], *r['ci95'], sci=True)} | {fmt_ci(d['mean'], *d['ci95'], sci=True)} | "
                f"{v['relative_change_grad']:+.1%} | {v['fraction_grad_lowers_J']:.0%} | {'yes' if met else 'no'} |")
        add("")
    g2 = report["networks"][gating]["gate2"]["pass"]
    g3 = audit["gate3"]["by_network"][gating]
    add(f"**Gate 2 ({gating}): {'PASS' if g2 else 'FAIL'}. Gate 3 ({gating}, logged states): {'PASS' if g3 else 'FAIL'}.** "
        + ("Phase C runs (amendment 11)." if g2 and g3 else "Stop (amendment 11): Phase C is not run."))
    add("")
    return {"gate2": g2, "gate3": g3, "collection": collection, "report": report, "audit": audit, "demo": demo, "gating": gating}


def pooled(episodes: pd.DataFrame, method: str) -> dict:
    t = episodes[(episodes["env"] == "target") & (episodes["method"] == method)]
    s = episodes[(episodes["env"] == "source") & (episodes["method"] == method)]
    return {"t200": t["success_200"].mean(), "t400": t["success_400"].mean(), "s200": s["success_200"].mean(), "s400": s["success_400"].mean(),
            "steps": t["steps"].mean(), "ee_mm": t["mean_ee_step_m"].mean() * 1e3}


def main() -> int:
    parser = ArgumentParser(description="Actuation run report (Phase B + Phase C)")
    parser.add_argument("--config", default="configs/sim2sim_pilot_actuation.yaml")
    parser.add_argument("--latency-config", default="configs/sim2sim_pilot.yaml")
    parser.add_argument("--out", default=None, help="default: <run>/<shift>/results.md")
    args = parser.parse_args()
    pilot = load_config(args.config)
    phase = pilot["phase_c"]
    shift_name = load_script("10_shift_scan.py").shift_from_entry(pilot["phase_b"]["shift"]).name
    root = Path(pilot["output_dir"]) / shift_name
    out = root / "phase_c"
    lines: list[str] = []
    add = lines.append

    add(f"# Sim-to-sim pilot: {shift_name} (Phase B and Phase C)")
    add("")
    add(f"Generated by `scripts/sim2sim/17_report_actuation.py` from `runs/sim2sim_pilot/{shift_name}/`. Spec: `docs/SIM2SIM_PILOT_SPEC.md` (amendments 1-18). "
        f"Grids, selection rules, the bootstrap and the gate operationalisations are in `{args.config}`, written before any actuation result existed.")
    add("")
    add(f"- Source: nominal ManiSkill PickCube-v1 (`physx_cpu`, `pd_ee_delta_pos`, state obs). Target: stiffness and damping of the 7 arm joint drives "
        f"scaled by s = {pilot['phase_b']['shift']['s']} (fingers untouched), applied once and verified after every reset and after every target restore. "
        "Chunk H = 20, e = 5, L = 15, K = 10, r = 0.")
    add("")
    b = phase_b_section(add, pilot, root, shift_name)
    if not (b["gate2"] and b["gate3"]) or not (out / "eval_episodes.parquet").is_file():
        path = Path(args.out) if args.out else root / "results.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 0

    methods = list(phase["methods"])
    boot = phase["bootstrap"]
    n_boot, bseed = int(boot["n_boot"]), int(boot["seed"])
    seeds = [int(s) for s in phase["seeds"]]
    selection = json.loads((out / "selection.json").read_text())
    summary = json.loads((out / "eval_summary.json").read_text())
    training = {s: json.loads((out / f"seed_{s}" / "training.json").read_text()) for s in seeds}
    episodes = pd.read_parquet(out / "eval_episodes.parquet")
    episodes["init"] = episodes["init_seed"]
    gaps = pd.read_parquet(out / "true_gap_sim2sim_only.parquet")
    sel_eps = pd.read_parquet(out / "selection_episodes.parquet")
    final = phase["final_eval"]
    init_seeds = list(range(final["seeds"]["start"], final["seeds"]["start"] + final["seeds"]["count"]))
    if sorted(episodes["init"].unique().tolist()) != init_seeds:
        raise RuntimeError("final evaluation initial states do not match the pre-registered seeds")

    add("## Phase C")
    add("")
    add(f"- Seeds {seeds}. Per seed a base policy (`02_prepare_policy.py` unchanged, per-seed config), then fine-tunes from it with the 06 recipe "
        "(5000 steps, lr 3e-4, batch 256, identical demonstration batch sequence per seed).")
    add(f"- Methods: {', '.join(SHORT[m] for m in methods)}. gap-logged is dropped (amendment 12). G = the sqrtJ network of this Phase B "
        "(`checkpoints/gap_sqrtJ.pt`), trained once from pi_bc's actuation-target rollouts and reused frozen for all seeds (amendment 18).")
    add("- Magnitude penalty: beta * mean over the H chunk steps and the 3 translation dims of the squared policy command (same scope and reduction "
        "as the smoothness penalty).")
    add(f"- Selection: every distinct candidate checkpoint runs in the target from the 100 val-split seeds (200-step cap); most successes wins, ties "
        f"to the smaller weight / earlier checkpoint. Final evaluation: seeds {init_seeds[0]}-{init_seeds[-1]} (never used before), target and source, "
        "one 400-step run per episode: success@200 (primary), success@400 (secondary).")
    add(f"- Intervals: two-way cluster bootstrap ({n_boot} resamples) over seeds and initial states (true gap: seeds and logged episodes); 95% percentile intervals.")
    add("")
    checks = {s: training[s].get("checks", {}) for s in seeds}
    add("Checks: imitation-selected's step-5000 checkpoint is bitwise the zero-shot policy for seeds "
        f"{[s for s in seeds if checks[s].get('imitation_last_checkpoint_equals_zero_shot')]} (it reuses zero-shot's selection rollouts); "
        "the magnitude trainer reproduces 06's smoothness fine-tune bitwise when given the smoothness term for seeds "
        f"{[s for s in seeds if checks[s].get('train_aux_reproduces_train_one_smooth', {}).get('bitwise_equal')]}.")
    add("")

    # ------------------------------------------------------------------ target data
    add("### Target data used")
    add("")
    tr, va = b["collection"]["train"], b["collection"]["val"]
    ft = training[seeds[0]]["finetune_data"]
    per_method = sel_eps[~sel_eps["shared_rollout"] | (sel_eps["method"] == "zero_shot")].groupby("method")["steps"].sum().to_dict()
    add("| Target steps | Count |")
    add("|---|---:|")
    add(f"| Labels for G, train logs (forward only) | {tr['target_steps_used']:,} |")
    add(f"| G validation logs (gates 2 and 3, forward only) | {va['target_steps_used']:,} |")
    add(f"| Selection, all methods and seeds ({selection['episodes']:,} episodes, {selection['distinct_checkpoints']} distinct checkpoints, {selection['aborted']} aborted) | {selection['target_steps_used']:,} |")
    for m in methods:
        add(f"| &nbsp;&nbsp;of which {SHORT[m]}{' (step 5000 = zero-shot, counted under zero-shot)' if m == 'imitation_selected' else ''} | {per_method.get(m, 0):,} |")
    add(f"| Final evaluation (not used for any choice) | {summary['target_steps_final_eval']:,} |")
    add("")
    add(f"Training data from the target: gap-demo uses G (the {tr['target_steps_used']:,} label steps); the same-budget fine-tune uses the {ft['episodes']} successful "
        f"logged train episodes ({ft['samples']:,} steps, {ft['perturbed_episodes']} perturbed); zero-shot, imitation-selected, smoothness and magnitude use none "
        "(but all except zero-shot are selected on target rollouts).")
    c = summary["consistency_check"]
    add("")
    add(f"Consistency check (400-step target env vs Phase A, pi_bc, 50 Phase A fresh seeds): {c['rerun_success_200']}/{c['episodes']} vs {c['phase_a_success']}/{c['episodes']} "
        f"successes within 200 steps; identical outcome per episode: {c['same_outcome_per_episode']}. Final evaluation: {summary['episodes']:,} episodes "
        f"({summary['target_steps_final_eval']:,} target and {summary['source_steps_final_eval']:,} source steps), {summary['aborted_episodes']} aborted.")
    add("")

    # ------------------------------------------------------------------ selected weights
    add("### Selected weights")
    add("")
    add("Target successes out of 100 val-split seeds per candidate; the selected one in bold.")
    add("")
    add("| Method | Weight | " + " | ".join(f"seed {s}" for s in seeds) + " |")
    add("|---|---|" + "---|" * len(seeds))
    for method in methods:
        cells = []
        for s in seeds:
            entry = selection["seeds"][str(s)][method]
            cells.append(", ".join(f"**{tag}: {n}**" if tag == entry["selected"] else f"{tag}: {n}" for tag, n in entry["candidates"].items()))
        add(f"| {LABELS[method]} | {WEIGHT[method]} | " + " | ".join(cells) + " |")
    add("")

    # ------------------------------------------------------------------ per seed + pooled
    gap_mean = gaps[gaps["status"] == "ok"].groupby(["seed", "method"])["true_gap_m2"].mean()
    add("### Per-seed results (final seeds, 150 initial states each)")
    add("")
    add("| Seed | Method | Selected | Target success@200 | Target success@400 | Source success@200 | Source success@400 | Target mean steps | Target mean EE step (mm) | True gap, own chunk (m²) |")
    add("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in seeds:
        for method in methods:
            t = episodes[(episodes["env"] == "target") & (episodes["seed"] == s) & (episodes["method"] == method)]
            src = episodes[(episodes["env"] == "source") & (episodes["seed"] == s) & (episodes["method"] == method)]
            add(f"| {s} | {SHORT[method]} | {selection['seeds'][str(s)][method]['selected']} | {int(t['success_200'].sum())}/150 | {int(t['success_400'].sum())}/150 | "
                f"{int(src['success_200'].sum())}/150 | {int(src['success_400'].sum())}/150 | {t['steps'].mean():.1f} | {t['mean_ee_step_m'].mean() * 1e3:.2f} | "
                f"{gap_mean.get((s, method), np.nan):.2e} |")
    add("")
    add("Pooled over seeds:")
    add("")
    add("| Method | Target success@200 | Target success@400 | Source success@200 | Source success@400 | Target mean steps | Target mean EE step (mm) | True gap (m²) |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    pooled_act = {}
    for method in methods:
        p = pooled(episodes, method)
        pooled_act[method] = p
        g = gaps[(gaps["method"] == method) & (gaps["status"] == "ok")]["true_gap_m2"].mean()
        add(f"| {LABELS[method]} | {p['t200']:.1%} | {p['t400']:.1%} | {p['s200']:.1%} | {p['s400']:.1%} | {p['steps']:.1f} | {p['ee_mm']:.2f} | {g:.2e} |")
    add("")

    # ------------------------------------------------------------------ paired differences
    target, source = episodes[episodes["env"] == "target"], episodes[episodes["env"] == "source"]
    gap_ok = gaps.copy()
    gap_ok["unit"] = gap_ok["episode_id"].astype(str) + ":" + gap_ok["t"].astype(str)
    unit_groups = gap_ok.drop_duplicates("unit").set_index("unit")["episode_id"]
    groups = unit_groups.loc[sorted(gap_ok["unit"].unique())].to_numpy()

    def paired(frame, a, bb, value, column="init", grp=None):
        return two_way_ci(matrix(frame, a, value, column) - matrix(frame, bb, value, column), grp, n_boot, bseed)

    results = {}
    add("### Pooled paired differences")
    add("")
    add("A − B over 5 seeds × 150 initial states (true gap: 5 seeds × 500 logged val states, clustered by episode).")
    add("")
    add("| A − B | Target success@200 (pp) | Target success@400 (pp) | Source success@200 (pp) | Source success@400 (pp) | Target mean EE step (mm) | Target mean steps | True gap (m²) |")
    add("|---|---|---|---|---|---|---|---|")
    for a, bb in phase["comparisons"]:
        r = {"t200": paired(target, a, bb, "success_200"), "t400": paired(target, a, bb, "success_400"),
             "s200": paired(source, a, bb, "success_200"), "s400": paired(source, a, bb, "success_400"),
             "ee": paired(target, a, bb, "mean_ee_step_m"), "steps": paired(target, a, bb, "steps"),
             "gap": paired(gap_ok, a, bb, "true_gap_m2", "unit", groups)}
        results[(a, bb)] = r
        add(f"| {SHORT[a]} − {SHORT[bb]} | {fmt_ci(*r['t200'], 100)} | {fmt_ci(*r['t400'], 100)} | {fmt_ci(*r['s200'], 100)} | {fmt_ci(*r['s400'], 100)} | "
            f"{fmt_ci(*r['ee'], 1e3, 2)} | {fmt_ci(*r['steps'], 1, 1)} | {fmt_ci(*r['gap'], sci=True)} |")
    add("")

    # ------------------------------------------------------------------ gates
    add("### Gates")
    add("")
    add(f"Gate 4 (pre-registered, one verdict per reference: {', '.join(SHORT[x] for x in pilot['gate4']['references'])}): {pilot['gate4']['rule']}")
    add("")
    add(f"Gate 5 (pre-registered): {pilot['gate5']['rule']}")
    add("")
    add("| Gate | Reference | Target success@200 difference (pp) | Source success@200 difference (pp) | Target mean EE step difference (mm) | Verdict |")
    add("|---|---|---|---|---|---|")
    verdicts = {}
    for ref in pilot["gate4"]["references"]:
        r = results[("gap_demo", ref)]
        reasons = []
        if not r["t200"][1] > 0:
            reasons.append("target success interval not above 0")
        if r["s200"][2] < 0:
            reasons.append("source success drops")
        if r["ee"][2] < 0:
            reasons.append("detectable slowdown")
        verdicts[f"gate4_vs_{ref}"] = (not reasons, reasons)
        add(f"| 4 | {SHORT[ref]} | {fmt_ci(*r['t200'], 100)} | {fmt_ci(*r['s200'], 100)} | {fmt_ci(*r['ee'], 1e3, 2)} | "
            f"**{'PASS' if not reasons else 'FAIL'}**{'' if not reasons else ' (' + '; '.join(reasons) + ')'} |")
    r5 = results[("gap_demo", "finetune")]
    g5 = not r5["t200"][2] < 0
    verdicts["gate5_vs_finetune"] = (g5, [] if g5 else ["interval entirely below 0"])
    add(f"| 5 | fine-tune | {fmt_ci(*r5['t200'], 100)} | {fmt_ci(*r5['s200'], 100)} | {fmt_ci(*r5['ee'], 1e3, 2)} | "
        f"**{'PASS' if g5 else 'FAIL'}**{'' if g5 else ' (interval entirely below 0)'} |")
    add("")
    add("Verdicts:")
    add("")
    add(f"- **Gate 2 (sqrtJ): {'PASS' if b['gate2'] else 'FAIL'}**")
    add(f"- **Gate 3 (sqrtJ, logged states): {'PASS' if b['gate3'] else 'FAIL'}**")
    for key, (ok, reasons) in verdicts.items():
        gate, ref = key.split("_vs_")
        add(f"- **Gate {gate[-1]}, gap-demo vs {SHORT[ref]}: {'PASS' if ok else 'FAIL'}**" + ("" if ok else f" — {'; '.join(reasons)}."))
    add("")
    add("Secondary (400-step cap, not a gate): " + "; ".join(
        f"gap-demo − {SHORT[ref]} {fmt_ci(*results[('gap_demo', ref)]['t400'], 100)} pp" for ref in list(pilot["gate4"]["references"]) + ["finetune"]) + ".")
    add("")

    # ------------------------------------------------------------------ G vs truth on own chunks
    import torch

    from chunk_trajectory_error.pipeline import ErrorBundle, PolicyBundle

    labels = dict(np.load(root / "labels_val.npz"))
    key_index = {(int(e), int(t)): i for i, (e, t) in enumerate(zip(labels["episode_id"], labels["t"]))}
    sample_idx = [key_index[(int(e), int(t))] for e, t in gaps.drop_duplicates(["episode_id", "t"])[["episode_id", "t"]].to_numpy()]
    obs, hist, robot = (torch.tensor(labels[k][sample_idx]) for k in ("obs", "history", "robot"))
    gap_net = ErrorBundle.load(phase["gap_checkpoint"])
    add("### G's prediction vs the true gap of each policy's own chunk (held-out logged states, sim-to-sim only)")
    add("")
    add("| Method | Mean G prediction (m²) | Mean true gap (m²) | True / predicted | Mean squared translation command | Translation entries at the action limit |")
    add("|---|---:|---:|---:|---:|---:|")
    for method in methods:
        preds, sat, mag = [], [], []
        for s in seeds:
            bundle = PolicyBundle.load(selection["seeds"][str(s)][method]["checkpoint"])
            with torch.no_grad():
                chunk = bundle(obs).clamp(-1.0, 1.0)[:, :10]
                preds.append(float(gap_net.cost_m2(hist, robot, chunk, 0).mean()))
                sat.append(float((chunk[..., :3].abs() > 0.999).float().mean()))
                mag.append(float((chunk[..., :3] ** 2).sum(-1).mean()))
        true_mean = gaps[(gaps["method"] == method) & (gaps["status"] == "ok")]["true_gap_m2"].mean()
        add(f"| {LABELS[method]} | {np.mean(preds):.2e} | {true_mean:.2e} | {true_mean / np.mean(preds):.1f} | {np.mean(mag):.3f} | {np.mean(sat):.1%} |")
    logged_actions = torch.tensor(labels["actions"][sample_idx])
    add(f"| (logged commanded actions, for reference) | {float(gap_net.cost_m2(hist, robot, logged_actions, 0).mean()):.2e} | "
        f"{float(labels['J_gap_m2'][sample_idx].mean()):.2e} | | {float((logged_actions[..., :3] ** 2).sum(-1).mean()):.3f} | "
        f"{float((np.abs(labels['actions'][sample_idx][..., :3]) > 0.999).mean()):.1%} |")
    add("")

    # ------------------------------------------------------------------ side by side with latency
    lat_pilot = load_config(args.latency_config)
    lat_root = Path(lat_pilot["output_dir"]) / load_script("10_shift_scan.py").shift_from_entry(lat_pilot["phase_c"]["shift"]).name
    lat_out = lat_root / "phase_c"
    if (lat_out / "paired_differences.json").is_file():
        lat_pd = json.loads((lat_out / "paired_differences.json").read_text())
        lat_eps = pd.read_parquet(lat_out / "eval_episodes.parquet")
        lat_sel = json.loads((lat_out / "selection.json").read_text())
        lat_collection = json.loads((lat_root / "collection_train.json").read_text())
        lat_report = json.loads((lat_root / "gap_report.json").read_text())
        lat_audit = json.loads((lat_root / "gap_audit_sim2sim_only.json").read_text())
        lat_demo = json.loads((lat_root / "gap_demo_diagnostic_sim2sim_only.json").read_text())
        add("## Side by side with latency d = 1")
        add("")
        add("Latency numbers are read from `runs/sim2sim_pilot/latency_d1/` (unchanged). Latency final seeds 200000-200149, actuation 300000-300149; "
            "latency had no imitation-selected or magnitude rows and a different lambda grid ({0.001 … 0.1}).")
        add("")
        add("| | Latency d = 1 | Actuation s = 0.4 |")
        add("|---|---|---|")
        add(f"| Successful logged train episodes (fine-tune data) | {lat_collection['episodes_success']} of {lat_collection['episodes']} | {tr['episodes_success']} of {tr['episodes']} |")
        add(f"| Train label median J (m²) | {lat_collection['J_gap_m2']['median']:.2e} | {tr['J_gap_m2']['median']:.2e} |")
        lv, av = lat_report["networks"]["sqrtJ"]["val"], b["report"]["networks"]["sqrtJ"]["val"]
        add(f"| Gate 2, sqrtJ: Spearman / MAE vs constant (m²) | {lv['spearman']:.2f} / {lv['MAE_m2']:.2e} vs {lv['constant_mean_MAE_m2']:.2e} | "
            f"{av['spearman']:.2f} / {av['MAE_m2']:.2e} vs {av['constant_mean_MAE_m2']:.2e} |")
        add(f"| Spearman true gap vs roughness (val) | {lat_report['roughness']['spearman_true_gap_vs_roughness_val']:.2f} | {b['report']['roughness']['spearman_true_gap_vs_roughness_val']:.2f} |")
        add(f"| Spearman true gap vs magnitude (val) | not computed | {b['report']['magnitude']['spearman_true_gap_vs_magnitude_val']:.2f} |")
        lt, at = lat_audit["test"]["sqrtJ"], b["audit"]["test"]["sqrtJ"]
        add(f"| Gate 3, sqrtJ, logged: relative ΔJ at chosen step | {lt['relative_change_grad']:+.1%} (step {lt['step_size']}) | {at['relative_change_grad']:+.1%} (step {at['step_size']}) |")
        if b["demo"]:
            ld, ad = lat_demo, b["demo"]
            add(f"| Demo states: G vs truth Spearman | {ld['prediction_vs_truth']['spearman']:.2f} | {ad['prediction_vs_truth']['spearman']:.2f} |")
            add(f"| Demo states: relative ΔJ at 0.02 / 0.05 / 0.1 | " + " / ".join(f"{ld['step_test'][k]['relative_change_grad']:+.1%}" for k in ld["step_test"]) + " | "
                + " / ".join(f"{ad['step_test'][k]['relative_change_grad']:+.1%}" for k in ad["step_test"]) + " |")
        add(f"| Target steps: labels + G validation | {lat_collection['target_steps_used']:,} + 5,000 | {tr['target_steps_used']:,} + {va['target_steps_used']:,} |")
        add(f"| Target steps: selection | {lat_sel['target_steps_used']:,} | {selection['target_steps_used']:,} |")
        for m in ("zero_shot", "gap_demo", "finetune", "smooth"):
            lp = pooled(lat_eps, m)
            ap = pooled_act[m]
            add(f"| {SHORT[m]}: target / source success@200 | {lp['t200']:.1%} / {lp['s200']:.1%} | {ap['t200']:.1%} / {ap['s200']:.1%} |")
        for m in ("imitation_selected", "magnitude"):
            ap = pooled_act[m]
            add(f"| {SHORT[m]}: target / source success@200 | - | {ap['t200']:.1%} / {ap['s200']:.1%} |")
        for a, bb in (("gap_demo", "zero_shot"), ("gap_demo", "finetune"), ("finetune", "zero_shot"), ("smooth", "zero_shot")):
            lr = lat_pd[f"{a}-{bb}"]
            add(f"| {SHORT[a]} − {SHORT[bb]}, target success@200 (pp) | {fmt_ci(*lr['t200'], 100)} | {fmt_ci(*results[(a, bb)]['t200'], 100)} |")
            add(f"| {SHORT[a]} − {SHORT[bb]}, source success@200 (pp) | {fmt_ci(*lr['s200'], 100)} | {fmt_ci(*results[(a, bb)]['s200'], 100)} |")
        lat_lambdas = [lat_sel["seeds"][str(s)]["gap_demo"]["selected"] for s in seeds]
        add(f"| gap-demo selected lambda per seed | {', '.join(t[1:] for t in lat_lambdas)} | "
            f"{', '.join(selection['seeds'][str(s)]['gap_demo']['selected'][1:] for s in seeds)} |")
        lv4 = lat_pd["verdicts"]
        add(f"| Gate 4, gap-demo vs zero-shot | {'PASS' if lv4['gate4_gap_demo']['pass'] else 'FAIL'} | {'PASS' if verdicts['gate4_vs_zero_shot'][0] else 'FAIL'} |")
        add(f"| Gate 4, gap-demo vs imitation-selected | - | {'PASS' if verdicts['gate4_vs_imitation_selected'][0] else 'FAIL'} |")
        add(f"| Gate 5, gap-demo vs fine-tune | {'PASS' if lv4['gate5_gap_demo']['pass'] else 'FAIL'} | {'PASS' if verdicts['gate5_vs_finetune'][0] else 'FAIL'} |")
        add("")

    # ------------------------------------------------------------------ notes
    add("## Notes")
    add("")
    add("- Zero-shot has a single candidate; every other method (including smoothness and magnitude, which use no target data for training) is "
        "selected on target rollouts. Imitation-selected isolates that selection effect for the imitation objective.")
    ratios = {}
    for s in seeds:
        for c in training[s]["candidates"]:
            log = c.get("last_log") or {}
            if c["method"] in ("gap_demo", "smooth", "magnitude") and log.get("grad_norm_imitation"):
                ratios.setdefault((c["method"], c["tag"]), []).append(log["grad_norm_weighted_auxiliary"] / log["grad_norm_imitation"])
    add("- Ratio of the weighted auxiliary gradient norm to the imitation gradient norm at step 5000 (mean over seeds): " + "; ".join(
        f"{SHORT[m]} {tag[1:]}: {np.mean(v):.2f}" for (m, tag), v in ratios.items()) + ". Gap-demo's smallest lambda already contributes "
        "about a tenth of the imitation gradient, while val BC loss is almost unchanged (`training.json`).")
    gd_selected = [selection["seeds"][str(s)]["gap_demo"]["selected"] for s in seeds]
    add(f"- gap-demo selected the lower edge of the lambda grid on {sum(t == 'w' + format(min(phase['lambdas']), 'g') for t in gd_selected)} of {len(seeds)} seeds "
        f"and magnitude the lower edge of its beta grid on {sum(selection['seeds'][str(s)]['magnitude']['selected'] == 'w' + format(min(phase['magnitude_betas']), 'g') for s in seeds)}; "
        "the grids were not changed after seeing this.")
    add("- Target mean steps and mean EE step are over the 400-step episodes (failures run to 400).")
    add("- G is trained once, from pi_bc's rollouts (seed-17 lineage), and reused for every seed.")
    add("- The true gap of each policy's own chunk and both gate-3 tests are sim-to-sim-only measurements (target restored); none is available on hardware.")
    path = Path(args.out) if args.out else root / "results.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_out = {f"{a}-{bb}": {k: list(v) for k, v in r.items()} for (a, bb), r in results.items()}
    json_out["verdicts"] = {k: {"pass": ok, "reasons": reasons} for k, (ok, reasons) in verdicts.items()}
    json_out["verdicts"]["gate2_sqrtJ"] = {"pass": b["gate2"]}
    json_out["verdicts"]["gate3_sqrtJ_logged"] = {"pass": b["gate3"]}
    if not args.out:
        (out / "paired_differences.json").write_text(json.dumps(json_out, indent=2), encoding="utf-8")
        manifest_path = Path(pilot["output_dir"]) / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault(pilot.get("manifest_section_c", "phase_c"), {})["17_report"] = {"results": str(path), "verdicts": json_out["verdicts"]}
        manifest_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
