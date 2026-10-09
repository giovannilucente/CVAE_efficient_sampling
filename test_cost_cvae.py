"""Check of a trained cost-aware CVAE on held-out planning cycles, through the inference interface.

For every cycle, K samples of CostAwareCVAE (CVAE.py) are compared with the cycle's feasible CEM
candidates, and with two samplers that ignore the scene: uniform in the cycle's search space, and
the overall target distribution (normal with the normalizer's mean / std).
    in_bounds    share of samples inside the cycle's search space
    best_dist    distance of the closest sample to the cycle's best CEM candidate (in std units)
    best_cost@K  J~ of the closest feasible candidate to the best sample (0 = CEM's best, 1 = J_90),
                 a proxy of the cost the planner would get from these K samples
    mean_dist    distance of the samples' mean to the cycle's target mean (the mean of the
                 distribution the CVAE is trained on), in std units: does the CVAE use the scene?
    spread       mean std of the samples per dimension (std units); compare with the spread of the
                 target distribution: a much smaller value means the CVAE collapsed to one point
A CVAE that uses the scene should beat both baselines; this is a code / sanity check, the planner's
closed-loop cost is measured with the planner itself.

python test_cost_cvae.py --run <run dir> [--cycles 500] [--k 16]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CVAE_efficient_sampling.CVAE import CostAwareCVAE  # noqa: E402
from CVAE_efficient_sampling.cem_dataset import CEMCache, select_cycles, split_by_scenario  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--cycles", type=int, default=500)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = CostAwareCVAE(args.run, device)
    cfg = sampler.config
    cache = CEMCache(cfg["val_cache"] or cfg["cache"])
    rows = select_cycles(cache.cycles, cfg["drop_tail_s"], keep_failed=not cfg["drop_failed"])
    if not cfg["val_cache"]:
        _, rows = split_by_scenario(cache.cycles, rows, cfg["val_fraction"], cfg["seed"])
    rng = np.random.default_rng(args.seed)
    rows = rng.permutation(rows)[:args.cycles]
    mean, std = sampler.normalizer.target_scaler.mean_, sampler.normalizer.target_scaler.scale_

    results = {name: {"in_bounds": [], "best_dist": [], "best_cost": [], "mean_dist": [], "spread": []}
               for name in ("cvae", "uniform", "marginal")}
    tau, correct = cfg["tau"], not cfg["no_density_correction"]
    target_spread = []
    for row in rows:
        r = cache.cycles.iloc[row]
        lo, hi = np.array([r.d_min, r.v_min, r.T_min]), np.array([r.d_max, r.v_max, r.T_max])
        s, n = r.cand_start, r.cand_count
        cand = np.asarray(cache.z[s:s + n], dtype=np.float64) / std
        jn = np.asarray(cache.jn[s:s + n])
        best = cand[np.argmin(jn)]
        logw = -jn / tau - (np.asarray(cache.logq[s:s + n]) if correct else 0.0)
        w = np.exp(logw - logw.max())
        target_mean = (w[:, None] * cand).sum(0) / w.sum()
        target_spread.append(np.sqrt((w[:, None] * (cand - target_mean) ** 2).sum(0) / w.sum()).mean())
        frames = [cache.images[r[f"frame_{h}"]] for h in range(3)]
        samples = {
            "cvae": sampler.generate_samples(frames, args.k),
            "uniform": lo + (hi - lo) * rng.random((args.k, 3)),
            "marginal": mean + std * rng.standard_normal((args.k, 3)),
        }
        for name, z in samples.items():
            res = results[name]
            res["in_bounds"].append(np.mean(np.all((z >= lo - 1e-6) & (z <= hi + 1e-6), axis=1)))
            zs = z / std
            res["best_dist"].append(np.min(np.linalg.norm(zs - best, axis=1)))
            nearest = np.argmin(((zs[:, None, :] - cand[None]) ** 2).sum(-1), axis=1)  # per sample
            res["best_cost"].append(np.min(jn[nearest]))
            res["mean_dist"].append(np.linalg.norm(zs.mean(0) - target_mean))
            res["spread"].append(zs.std(0).mean())

    print(f"{len(rows)} held-out cycles, K = {args.k} samples each (run {args.run})")
    print(f"{'sampler':10s} {'in_bounds':>10s} {'best_dist':>10s} {'best_cost@K':>12s} {'mean_dist':>10s} {'spread':>8s}")
    for name, res in results.items():
        print(f"{name:10s} {np.mean(res['in_bounds']):10.3f} {np.median(res['best_dist']):10.3f} "
              f"{np.median(res['best_cost']):12.3f} {np.median(res['mean_dist']):10.3f} {np.mean(res['spread']):8.3f}")
    print(f"target distribution (what the CVAE should reproduce): spread {np.mean(target_spread):.3f}")


if __name__ == "__main__":
    main()
