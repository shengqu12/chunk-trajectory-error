"""Phase C, step 3: ``runs/sim2sim_pilot/<shift>/results.md``.

Pooled paired differences use a two-way cluster bootstrap: seeds and initial states (or logged
episodes, for the true gap) are resampled independently; 95% percentile intervals.  The gate rules
are read from ``configs/sim2sim_pilot.yaml`` (pre-registered before any Phase C result).
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


METHODS = ("zero_shot", "gap_demo", "gap_logged", "finetune", "smooth")
LABELS = {"zero_shot": "Zero-shot (imitation only)", "gap_demo": "Gap penalty, G on demo states (gap-demo)",
          "gap_logged": "Gap penalty, G on logged target states (gap-logged)", "finetune": "Same-budget fine-tune",
          "smooth": "Action smoothness"}
WEIGHT = {"zero_shot": "-", "gap_demo": "lambda", "gap_logged": "lambda", "finetune": "checkpoint step", "smooth": "beta"}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name[:-3].lstrip("0123456789_"), _script_path(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def two_way_ci(values: np.ndarray, groups: np.ndarray | None, n_boot: int, seed: int) -> tuple[float, float, float]:
    """values [S, N]; rows (seeds) and column groups resampled independently; mean with 95% interval."""
    values = np.asarray(values, np.float64)
    S, N = values.shape
    groups = np.arange(N) if groups is None else np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    row_counts = np.stack([np.bincount(r, minlength=S) for r in rng.integers(0, S, (n_boot, S))])
    group_counts = np.stack([np.bincount(g, minlength=len(unique)) for g in rng.integers(0, len(unique), (n_boot, len(unique)))])
    col_counts = group_counts[:, inverse]
    stats = np.einsum("bs,sn,bn->b", row_counts, values, col_counts) / (S * col_counts.sum(1))
    return float(values.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def matrix(frame: pd.DataFrame, method: str, value: str, column: str = "init") -> np.ndarray:
    sub = frame[frame["method"] == method].pivot(index="seed", columns=column, values=value).sort_index()
    if sub.isna().any().any():
        raise RuntimeError(f"missing cells for {method}/{value}")
    return sub.to_numpy(np.float64)


def fmt_ci(mean: float, low: float, high: float, scale: float = 1.0, digits: int = 1, sci: bool = False) -> str:
    if sci:
        return f"{mean * scale:+.2e} [{low * scale:+.2e}, {high * scale:+.2e}]"
    return f"{mean * scale:+.{digits}f} [{low * scale:+.{digits}f}, {high * scale:+.{digits}f}]"


def main() -> int:
    parser = ArgumentParser(description="Phase C report")
    parser.add_argument("--config", default="configs/sim2sim_pilot.yaml")
    args = parser.parse_args()
    pilot = load_config(args.config)
    phase = pilot["phase_c"]
    boot = phase["bootstrap"]
    shift_name = load_script("10_shift_scan.py").shift_from_entry(phase["shift"]).name
    root = Path(pilot["output_dir"]) / shift_name
    out = root / "phase_c"
    seeds = [int(s) for s in phase["seeds"]]
    selection = json.loads((out / "selection.json").read_text())
    summary = json.loads((out / "eval_summary.json").read_text())
    diag = json.loads((root / "gap_demo_diagnostic_sim2sim_only.json").read_text())
    gap_report = json.loads((root / "gap_report.json").read_text())
    collection = {s: json.loads((root / f"collection_{s}.json").read_text()) for s in ("train", "val")}
    training = {s: json.loads((out / f"seed_{s}" / "training.json").read_text()) for s in seeds}
    episodes = pd.read_parquet(out / "eval_episodes.parquet")
    episodes["init"] = episodes["init_seed"]  # "seed" is the policy seed (job tag)
    gaps = pd.read_parquet(out / "true_gap_sim2sim_only.parquet")
    sel_eps = pd.read_parquet(out / "selection_episodes.parquet")
    return write_report(pilot, phase, boot, shift_name, root, out, seeds, selection, summary, diag, gap_report, collection,
                        training, episodes, gaps, sel_eps)


def write_report(pilot, phase, boot, shift_name, root, out, seeds, selection, summary, diag, gap_report, collection,
                 training, episodes, gaps, sel_eps) -> int:
    final = phase["final_eval"]
    init_seeds = list(range(final["seeds"]["start"], final["seeds"]["start"] + final["seeds"]["count"]))
    if sorted(episodes["init"].unique().tolist()) != init_seeds:
        raise RuntimeError("final evaluation initial states do not match the pre-registered seeds")
    n_boot, bseed = int(boot["n_boot"]), int(boot["seed"])
    lines: list[str] = []
    add = lines.append

    add(f"# Sim-to-sim pilot, Phase C: {shift_name}")
    add("")
    add("Generated by `scripts/sim2sim/16_report_sim2sim.py` from `runs/sim2sim_pilot/latency_d1/phase_c/`. Spec: `docs/SIM2SIM_PILOT_SPEC.md` "
        "(including the amendments of 2026-11-13). All weights, grids, selection rules, the bootstrap and the gate "
        "operationalisations were written to `configs/sim2sim_pilot.yaml` before any Phase C result existed.")
    add("")
    add("## Setup")
    add("")
    add(f"- Source: nominal ManiSkill PickCube-v1 (`physx_cpu`, `pd_ee_delta_pos`, state obs). Target: the same with a one-step command "
        f"latency (d = 1; first d steps hold with the first command's gripper value). Chunk H = 20, e = 5, L = 15, K = 10, r = 0.")
    add(f"- Seeds {seeds}. Per seed a base policy (`02_prepare_policy.py` unchanged, per-seed config), then five fine-tunes from it with the "
        "06 recipe (5000 steps, lr 3e-4, batch 256, identical demonstration batch sequence per seed).")
    add("- G: the sqrtJ gap network of Phase B (`checkpoints/gap_sqrtJ.pt`), trained **once** from pi_bc's target rollouts "
        "(pi_bc = the seed-17 BC fine-tune of `runs/maniskill_pickcube_official`) and reused frozen for all seeds.")
    add(f"- Selection: every candidate is run in the target from the 100 val-split seeds (200-step cap); most successes wins, ties to the "
        f"smaller weight / earlier checkpoint. Final evaluation: seeds {init_seeds[0]}-{init_seeds[-1]} (never used before), target and source, "
        "one 400-step run per episode: success@200 (primary) = success within 200 steps, success@400 (secondary).")
    add(f"- Intervals: two-way cluster bootstrap ({n_boot} resamples) over seeds and initial states (true gap: seeds and logged episodes); "
        "95% percentile intervals of the pooled paired mean difference.")
    add("")
    add("### Target data used")
    add("")
    tr, va = collection["train"], collection["val"]
    sel_steps = sel_eps.groupby("method")["steps"].sum().to_dict()
    sel_eps_n = sel_eps.groupby("method").size().to_dict()
    ft = training[seeds[0]]["finetune_data"]
    add("| Method | Training data from the target | Target steps in selection (all seeds) |")
    add("|---|---|---:|")
    add(f"| {LABELS['zero_shot']} | none | {sel_steps.get('zero_shot', 0):,} (reported only, nothing selected) |")
    add(f"| {LABELS['gap_demo']} | G: {tr['target_steps_used']:,} logged steps (labels) + {va['target_steps_used']:,} (G validation) | {sel_steps.get('gap_demo', 0):,} |")
    add(f"| {LABELS['gap_logged']} | same G + the {tr['label_samples_ok']:,} logged train states as inputs | {sel_steps.get('gap_logged', 0):,} |")
    add(f"| {LABELS['finetune']} | the {ft['episodes']} successful episodes ({ft['samples']:,} steps, {ft['perturbed_episodes']} perturbed) among the {tr['target_steps_used']:,} logged train steps | {sel_steps.get('finetune', 0):,} |")
    add(f"| {LABELS['smooth']} | none | {sel_steps.get('smooth', 0):,} |")
    add("")
    add(f"Selection: {selection['episodes']:,} target episodes, {selection['target_steps_used']:,} target steps in total, {selection['aborted']} aborted. "
        f"Final evaluation: {summary['episodes']:,} episodes ({summary['target_steps_final_eval']:,} target and {summary['source_steps_final_eval']:,} source steps), "
        f"{summary['aborted_episodes']} aborted.")
    c = summary["consistency_check"]
    add(f"Consistency check (400-step target env vs Phase A, pi_bc, 50 Phase A fresh seeds): {c['rerun_success_200']}/{c['episodes']} vs "
        f"{c['phase_a_success']}/{c['episodes']} successes within 200 steps; identical outcome per episode: {c['same_outcome_per_episode']}; "
        f"identical step counts: {c['same_steps_where_success']}.")
    add("")

    # ------------------------------------------------------------------ diagnostic
    add("## Pre-training diagnostic: G on demonstration states (sim-to-sim only, non-gating)")
    add("")
    p = diag["prediction_vs_truth"]
    add(f"{diag['samples']} samples from {diag['episodes']} held-out train demonstrations (train-split episodes whose seeds were not used for "
        "the Phase B target logs); pi_bc's chunk at the demonstration observation; source and target restored to the demonstration state, "
        "target pending command = previous demonstration action.")
    add("")
    add("| | p10 | median | p90 | mean |")
    add("|---|---:|---:|---:|---:|")
    for key, label in (("true_gap_demo_states_m2", "True gap, demo states (m²)"), ("G_pred_demo_states_m2", "G prediction, demo states (m²)"),
                       ("labels_logged_val_m2", "Labels, logged val states (m²)")):
        d = diag[key]
        add(f"| {label} | {d['p10']:.2e} | {d['median']:.2e} | {d['p90']:.2e} | {d['mean']:.2e} |")
    add("")
    add(f"G vs truth on demo states: Spearman {p['spearman']:.2f}, Pearson {p['pearson']:.2f}, MAE {p['MAE_m2']:.2e} m² "
        f"(constant = G's train mean: {p['constant_fit_mean_MAE_m2']:.2e}; constant = demo mean: {p['constant_demo_mean_MAE_m2']:.2e}); "
        f"mean prediction / mean truth {p['mean_pred_over_mean_true']:.2f}.")
    add("")
    add("| Step | ΔJ gradient (m²) [95% CI] | ΔJ random (m²) [95% CI] | Gradient − random (m²) [95% CI] | Relative ΔJ (gradient) | Gradient lowers J |")
    add("|---:|---|---|---|---:|---:|")
    for eta, v in diag["step_test"].items():
        g, q, d = v["delta_J_grad_m2"], v["delta_J_rand_m2"], v["paired_grad_minus_rand_m2"]
        add(f"| {eta} | {fmt_ci(g['mean'], *g['ci95'], sci=True)} | {fmt_ci(q['mean'], *q['ci95'], sci=True)} | {fmt_ci(d['mean'], *d['ci95'], sci=True)} | "
            f"{v['relative_change_grad']:+.1%} | {v['fraction_grad_lowers_J']:.0%} |")
    add("")
    add(f"Intervals cluster by demonstration episode. Skipped: {diag['skipped']}.")
    add("")

    # ------------------------------------------------------------------ selection
    add("## Selected weights")
    add("")
    add("Target successes out of 100 val-split seeds for every candidate; the selected one is in bold.")
    add("")
    add("| Method | Weight | " + " | ".join(f"seed {s}" for s in seeds) + " |")
    add("|---|---|" + "---|" * len(seeds))
    for method in METHODS:
        cells = []
        for s in seeds:
            entry = selection["seeds"][str(s)][method]
            cells.append(", ".join(f"**{tag}: {n}**" if tag == entry["selected"] else f"{tag}: {n}" for tag, n in entry["candidates"].items()))
        add(f"| {LABELS[method]} | {WEIGHT[method]} | " + " | ".join(cells) + " |")
    add("")

    # ------------------------------------------------------------------ per-seed
    gap_mean = gaps[gaps["status"] == "ok"].groupby(["seed", "method"])["true_gap_m2"].mean()
    add("## Per-seed results (final seeds, 150 initial states each)")
    add("")
    add("| Seed | Method | Selected | Target success@200 | Target success@400 | Source success@200 | Source success@400 | Target mean steps | Target mean EE step (mm) | True gap, own chunk (m²) |")
    add("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in seeds:
        for method in METHODS:
            row = []
            for env in ("target", "source"):
                sub = episodes[(episodes["env"] == env) & (episodes["seed"] == s) & (episodes["method"] == method)]
                row.append(sub)
            t, src = row
            add(f"| {s} | {LABELS[method]} | {selection['seeds'][str(s)][method]['selected']} | {int(t['success_200'].sum())}/150 | {int(t['success_400'].sum())}/150 | "
                f"{int(src['success_200'].sum())}/150 | {int(src['success_400'].sum())}/150 | {t['steps'].mean():.1f} | {t['mean_ee_step_m'].mean() * 1e3:.2f} | "
                f"{gap_mean.get((s, method), np.nan):.2e} |")
    add("")
    add("Pooled over seeds:")
    add("")
    add("| Method | Target success@200 | Target success@400 | Source success@200 | Source success@400 | Target mean steps | Target mean EE step (mm) | True gap (m²) |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for method in METHODS:
        t = episodes[(episodes["env"] == "target") & (episodes["method"] == method)]
        src = episodes[(episodes["env"] == "source") & (episodes["method"] == method)]
        g = gaps[(gaps["method"] == method) & (gaps["status"] == "ok")]["true_gap_m2"].mean()
        add(f"| {LABELS[method]} | {t['success_200'].mean():.1%} | {t['success_400'].mean():.1%} | {src['success_200'].mean():.1%} | "
            f"{src['success_400'].mean():.1%} | {t['steps'].mean():.1f} | {t['mean_ee_step_m'].mean() * 1e3:.2f} | {g:.2e} |")
    add("")

    # ------------------------------------------------------------------ paired differences
    target, source = episodes[episodes["env"] == "target"], episodes[episodes["env"] == "source"]
    gap_ok = gaps.copy()
    gap_ok["unit"] = gap_ok["episode_id"].astype(str) + ":" + gap_ok["t"].astype(str)
    unit_groups = gap_ok.drop_duplicates("unit").set_index("unit")["episode_id"]

    def paired(frame, a, b, value, column="init", groups=None):
        return two_way_ci(matrix(frame, a, value, column) - matrix(frame, b, value, column), groups, n_boot, bseed)

    comparisons = [("gap_demo", "zero_shot"), ("gap_logged", "zero_shot"), ("finetune", "zero_shot"), ("smooth", "zero_shot"),
                   ("gap_demo", "finetune"), ("gap_logged", "finetune"), ("gap_demo", "smooth"), ("gap_logged", "smooth")]
    results = {}
    add("## Pooled paired differences")
    add("")
    add("A − B over 5 seeds × 150 initial states (true gap: 5 seeds × 500 logged states clustered by episode).")
    add("")
    add("| A − B | Target success@200 (pp) | Target success@400 (pp) | Source success@200 (pp) | Source success@400 (pp) | Target mean EE step (mm) | Target mean steps | True gap (m²) |")
    add("|---|---|---|---|---|---|---|---|")
    units = sorted(gap_ok["unit"].unique())
    groups = unit_groups.loc[units].to_numpy()
    for a, b in comparisons:
        r = {
            "t200": paired(target, a, b, "success_200"), "t400": paired(target, a, b, "success_400"),
            "s200": paired(source, a, b, "success_200"), "s400": paired(source, a, b, "success_400"),
            "ee": paired(target, a, b, "mean_ee_step_m"), "steps": paired(target, a, b, "steps"),
            "gap": paired(gap_ok, a, b, "true_gap_m2", "unit", groups),
        }
        results[(a, b)] = r
        add(f"| {a} − {b} | {fmt_ci(*r['t200'], 100)} | {fmt_ci(*r['t400'], 100)} | {fmt_ci(*r['s200'], 100)} | {fmt_ci(*r['s400'], 100)} | "
            f"{fmt_ci(*r['ee'], 1e3, 2)} | {fmt_ci(*r['steps'], 1, 1)} | {fmt_ci(*r['gap'], sci=True)} |")
    add("")

    # ------------------------------------------------------------------ gates
    add("## Gates")
    add("")
    add(f"Gate 4 (pre-registered): {pilot['gate4']['rule']}")
    add("")
    add(f"Gate 5 (pre-registered): {pilot['gate5']['rule']}")
    add("")
    add("| Variant | Gate | Target success@200 difference (pp) | Source success@200 difference (pp) | Target mean EE step difference (mm) | Verdict |")
    add("|---|---|---|---|---|---|")
    verdicts = {}
    for variant in ("gap_demo", "gap_logged"):
        r = results[(variant, "zero_shot")]
        improves = r["t200"][1] > 0
        no_source_drop = not r["s200"][2] < 0
        no_slowdown = not r["ee"][2] < 0
        g4 = improves and no_source_drop and no_slowdown
        reasons = []
        if not improves:
            reasons.append("target success interval not above 0")
        if not no_source_drop:
            reasons.append("source success drops")
        if not no_slowdown:
            reasons.append("detectable slowdown")
        verdicts[(variant, 4)] = (g4, reasons)
        add(f"| {variant} | 4 (vs zero-shot) | {fmt_ci(*r['t200'], 100)} | {fmt_ci(*r['s200'], 100)} | {fmt_ci(*r['ee'], 1e3, 2)} | "
            f"**{'PASS' if g4 else 'FAIL'}**{'' if g4 else ' (' + '; '.join(reasons) + ')'} |")
        r5 = results[(variant, "finetune")]
        g5 = not r5["t200"][2] < 0
        verdicts[(variant, 5)] = (g5, [] if g5 else ["interval entirely below 0"])
        add(f"| {variant} | 5 (vs same-budget fine-tune) | {fmt_ci(*r5['t200'], 100)} | {fmt_ci(*r5['s200'], 100)} | {fmt_ci(*r5['ee'], 1e3, 2)} | "
            f"**{'PASS' if g5 else 'FAIL'}**{'' if g5 else ' (interval entirely below 0)'} |")
    add("")
    add("Verdicts:")
    add("")
    for variant in ("gap_demo", "gap_logged"):
        for gate in (4, 5):
            ok, reasons = verdicts[(variant, gate)]
            add(f"- **Gate {gate}, {variant}: {'PASS' if ok else 'FAIL'}**" + ("" if ok else f" — {'; '.join(reasons)}."))
    add("")
    add("Secondary (400-step cap, not a gate): " + "; ".join(
        f"{v} − zero-shot {fmt_ci(*results[(v, 'zero_shot')]['t400'], 100)} pp, {v} − fine-tune {fmt_ci(*results[(v, 'finetune')]['t400'], 100)} pp"
        for v in ("gap_demo", "gap_logged")) + ".")
    add("")

    # ------------------------------------------------------------------ G vs truth on the policies' own chunks
    import torch

    from chunk_trajectory_error.pipeline import ErrorBundle, PolicyBundle

    labels = dict(np.load(root / "labels_val.npz"))
    key_index = {(int(e), int(t)): i for i, (e, t) in enumerate(zip(labels["episode_id"], labels["t"]))}
    sample_idx = [key_index[(int(e), int(t))] for e, t in gaps.drop_duplicates(["episode_id", "t"])[["episode_id", "t"]].to_numpy()]
    obs, hist, robot = (torch.tensor(labels[k][sample_idx]) for k in ("obs", "history", "robot"))
    gap_net = ErrorBundle.load(phase["gap_checkpoint"])
    add("## G's prediction vs the true gap of each policy's own chunk (held-out logged states, sim-to-sim only)")
    add("")
    add("Same 500 logged val states as the true-gap column; chunk = first K actions at the logged observation.")
    add("")
    add("| Method | Mean G prediction (m²) | Mean true gap (m²) | True / predicted | Translation entries at the action limit |")
    add("|---|---:|---:|---:|---:|")
    for method in METHODS:
        preds, sat = [], []
        for s in seeds:
            bundle = PolicyBundle.load(selection["seeds"][str(s)][method]["checkpoint"])
            with torch.no_grad():
                chunk = bundle(obs).clamp(-1.0, 1.0)[:, :10]
                preds.append(float(gap_net.cost_m2(hist, robot, chunk, 0).mean()))
                sat.append(float((chunk[..., :3].abs() > 0.999).float().mean()))
        true_mean = gaps[(gaps["method"] == method) & (gaps["status"] == "ok")]["true_gap_m2"].mean()
        add(f"| {LABELS[method]} | {np.mean(preds):.2e} | {true_mean:.2e} | {true_mean / np.mean(preds):.1f} | {np.mean(sat):.1%} |")
    add(f"| (logged commanded actions, for reference) | {float(gap_net.cost_m2(hist, robot, torch.tensor(labels['actions'][sample_idx]), 0).mean()):.2e} | "
        f"{float(labels['J_gap_m2'][sample_idx].mean()):.2e} | | {float((np.abs(labels['actions'][sample_idx][..., :3]) > 0.999).mean()):.1%} |")
    add("")

    # ------------------------------------------------------------------ notes
    add("## Notes")
    add("")
    diag_path = out / "consistency_diagnosis.json"
    if diag_path.is_file():
        cd = json.loads(diag_path.read_text())
        add(f"- Consistency check: in a fresh environment running only those 50 seeds, the 200-step and 400-step caps give identical outcomes per "
            f"episode ({cd['success_200cap']} vs {cd['success_400cap_within_200']}; cap200 == cap400: {cd['cap200_equals_cap400_per_episode']}) and "
            f"repeats are bit-identical ({cd['repeat_bit_identical']}). The {cd['episodes_differing_from_phase_a']} episode that differs from Phase A "
            "comes from the environment's history: Phase A ran 100 test-episode resets before these seeds, and PhysX carries solver state across "
            "resets. This is noise across evaluation orders, not an effect of the 400-step cap (`phase_c/consistency_diagnosis.py`).")
    add("- Every method with a weight or checkpoint choice (gap-demo, gap-logged, fine-tune, smoothness) was selected on target success over "
        "the 100 val seeds; zero-shot had a single candidate. Differences against zero-shot therefore include the benefit of target-based selection.")
    add("- Target mean steps and mean EE step are over the 400-step episodes (failures run to 400).")
    add(f"- G is trained once, from pi_bc's rollouts (seed 17 lineage), and reused for every seed; the other seeds' base policies never produced target data.")
    add(f"- Gap-network quality (Phase B, logged val states): Spearman {gap_report['networks']['sqrtJ']['val']['spearman']:.2f}, "
        f"MAE {gap_report['networks']['sqrtJ']['val']['MAE_m2']:.2e} m² vs constant {gap_report['networks']['sqrtJ']['val']['constant_mean_MAE_m2']:.2e} m².")
    add("- Selection uses target rollouts from the val-split seeds for every method that has a weight, including smoothness (which uses no "
        "target data for training); the counts above are those selection steps.")
    add("- The true gap of each policy's own chunk is a sim-to-sim-only measurement (target restored from the Phase B val logs, including the "
        "pending command); it is not available on hardware.")
    (root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_out = {f"{a}-{b}": {k: list(v) for k, v in r.items()} for (a, b), r in results.items()}
    json_out["verdicts"] = {f"gate{g}_{v}": {"pass": ok, "reasons": reasons} for (v, g), (ok, reasons) in verdicts.items()}
    (out / "paired_differences.json").write_text(json.dumps(json_out, indent=2), encoding="utf-8")
    manifest_path = Path(pilot["output_dir"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("phase_c", {})["16_report"] = {"results": str(root / "results.md"), "verdicts": json_out["verdicts"]}
    manifest_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
