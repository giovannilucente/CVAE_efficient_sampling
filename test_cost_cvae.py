"""Check of a trained cost-aware CVAE on held-out planning cycles, through the inference interface.

K samples of CostAwareCVAE (CVAE.py) per cycle are compared with two samplers that ignore the
scene: uniform in the cycle's search space, and the overall target distribution (normal with the
normalizer's mean / std). Metrics: see generation_check.py. A CVAE that uses the scene should beat
both baselines in mean_dist and best_cost@K, with a spread close to the target spread. This is a
code / sanity check; the planner's closed-loop cost is measured with the planner itself.

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
from CVAE_efficient_sampling.generation_check import generation_metrics  # noqa: E402


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
    rows = np.random.default_rng(args.seed).permutation(rows)[:args.cycles]
    mean, std = sampler.normalizer.target_scaler.mean_, sampler.normalizer.target_scaler.scale_
    samplers = {
        "cvae": lambda frames, lo, hi, k, rng: sampler.generate_samples(frames, k),
        "uniform": lambda frames, lo, hi, k, rng: lo + (hi - lo) * rng.random((k, 3)),
        "marginal": lambda frames, lo, hi, k, rng: mean + std * rng.standard_normal((k, 3)),
    }
    summary = generation_metrics(samplers, cache, rows, args.k, cfg["tau"], not cfg["no_density_correction"],
                                 std, seed=args.seed)

    print(f"{len(rows)} held-out cycles, K = {args.k} samples each (run {args.run})")
    print(f"{'sampler':10s} {'in_bounds':>10s} {'best_dist':>10s} {'best_cost@K':>12s} {'mean_dist':>10s} {'spread':>8s}")
    for name in samplers:
        m = summary[name]
        print(f"{name:10s} {m['in_bounds']:10.3f} {m['best_dist']:10.3f} {m['best_cost']:12.3f} "
              f"{m['mean_dist']:10.3f} {m['spread']:8.3f}")
    print(f"target distribution (what the CVAE should reproduce): spread {summary['target_spread']:.3f}")
    log = os.path.join(args.run, "log.tsv")
    if os.path.exists(log):
        print("per-epoch check during training (log.tsv):")
        print(open(log).read())


if __name__ == "__main__":
    main()
