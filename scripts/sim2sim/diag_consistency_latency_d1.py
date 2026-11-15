"""Diagnostic for latency d = 1; writes runs/sim2sim_pilot/latency_d1/phase_c/consistency_diagnosis.json.

Why the 15 consistency check differs from Phase A: cap length vs environment history (pi_bc, target, 50 Phase A fresh seeds)."""
import json
import pandas as pd
from chunk_trajectory_error.sim2sim_jobs import closed_loop_worker, run_pool


def main():
    P = "configs/sim2sim_pilot.yaml"
    pol = "runs/maniskill_pickcube_official/checkpoints/pi_bc/pi.pt"
    seeds = list(range(100000, 100050))
    job = lambda: [{"policy_path": pol, "seeds": seeds, "tags": {}}]
    res = run_pool(closed_loop_worker, [(P, "target", 200, job()), (P, "target", 400, job()), (P, "target", 200, job())], 3)
    a = pd.read_parquet("runs/sim2sim_pilot/shift_scan_episodes.parquet")
    a = a[(a.condition == "latency_d1") & (a.init_source == "fresh_seed")].sort_values("seed")
    fr = [pd.DataFrame(r).sort_values("init_seed") for r in res]
    s200, s400, s200b = [f.success.values & (f.steps.values <= 200) for f in fr]
    out = {"note": "fresh environment, only these 50 seeds; Phase A ran 100 test-episode resets (with init states) before them",
           "success_200cap": int(s200.sum()), "success_400cap_within_200": int(s400.sum()), "success_200cap_repeat": int(s200b.sum()),
           "phase_a": int(a.success.sum()),
           "cap200_equals_cap400_per_episode": bool((s200 == s400).all()),
           "repeat_bit_identical": bool((s200 == s200b).all() and (fr[0].steps.values == fr[2].steps.values).all()),
           "equals_phase_a_per_episode": bool((s200 == a.success.values).all()),
           "episodes_differing_from_phase_a": int((s200 != a.success.values).sum())}
    json.dump(out, open("runs/sim2sim_pilot/latency_d1/phase_c/consistency_diagnosis.json", "w"), indent=2)
    print(out)


if __name__ == "__main__":
    main()
