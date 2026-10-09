"""Cost-aware CVAE training on the CEM data set (cache from cem_cache.py).

Each item is a planning cycle with one feasible CEM candidate z = [d, v, T] drawn with probability
proportional to exp(-J~ / tau) / q(z) (see cem_dataset.py); the loss is the CVAE's ELBO.

Long trainings run as several jobs (e.g. 24 h HPC limit): a checkpoint is written every
--ckpt_minutes and at the end of every epoch, the run stops cleanly after --max_hours or on
SIGTERM / SIGUSR1 (sent by Slurm before the time limit with --signal), and starting the same
command again resumes from the last checkpoint, at the same position of the epoch.

python train_cost_cvae.py --cache <cache dir> --out <run dir> [--model attn|hcvae] [--tau 0.5] ...
Outputs in <run dir>: ckpt_last.pt, model_best.pth (state dict), normalizer/, config.json, log.tsv
"""
import argparse
import json
import logging
import os
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beta_annealer import BetaAnnealer  # noqa: E402
from cem_dataset import CEMCache, CEMCycleDataset, ResumableSampler, select_cycles, split_by_scenario  # noqa: E402
from normalizer import Normalizer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True, help="training cache (cem_cache.py)")
    p.add_argument("--val_cache", help="validation cache; default: --val_fraction of the training scenarios")
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--out", required=True, help="run directory (resumed if it holds ckpt_last.pt)")
    p.add_argument("--model", choices=("attn", "hcvae"), default="attn")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=64)
    # cost-aware targets
    p.add_argument("--tau", type=float, default=0.5, help="temperature of exp(-J~ / tau)")
    p.add_argument("--no_density_correction", action="store_true", help="do not divide by CEM's density q(z)")
    p.add_argument("--draws_per_cycle", type=int, default=1, help="items per planning cycle and epoch")
    p.add_argument("--drop_tail_s", type=float, default=2.5, help="dropped seconds before a non-rear-end failure")
    p.add_argument("--drop_failed", action="store_true", help="drop non-rear-end failed scenarios entirely")
    # optimisation
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_milestones", type=int, nargs="*", default=[5])
    p.add_argument("--beta_end", type=float, default=1.0, help="final KL weight")
    p.add_argument("--beta_schedule", default="sigmoid", choices=("sigmoid", "linear", "constant"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_max_cycles", type=int, default=20000, help="validation subset size (fixed)")
    p.add_argument("--limit_scenarios", type=int, help="only the first n scenarios of the cache (smoke tests)")
    # job control
    p.add_argument("--max_hours", type=float, default=23.5, help="stop and checkpoint after this wall time")
    p.add_argument("--ckpt_minutes", type=float, default=30.0)
    return p.parse_args()


def build_model(args):
    if args.model == "attn":
        from attn_cvae import attnCVAE, reconstruction_kld
        model = attnCVAE(hidden_dim=32, input_dim=3, img_channels=3, img_size=args.img_size, latent_dim=args.latent_dim)
    else:
        from hcvae import HierarchicalCVAE, reconstruction_kld
        model = HierarchicalCVAE(hidden_dim=32, input_dim=3, img_channels=3, img_size=args.img_size,
                                 latent_dim=args.latent_dim, attn=True)
    return model, reconstruction_kld


class StopRequest:
    """Set by SIGTERM / SIGUSR1 (Slurm --signal) to checkpoint and exit at the next batch."""

    def __init__(self):
        self.requested = False
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, self._handler)

    def _handler(self, signum, frame):
        logging.info(f"signal {signum}: checkpoint and stop after this batch")
        self.requested = True


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(args.out, "train.log"))])
    t_start = time.time()
    stop = StopRequest()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(args.out, "ckpt_last.pt")
    resume = torch.load(ckpt_path, map_location="cpu", weights_only=False) if os.path.exists(ckpt_path) else None
    if resume:
        saved = resume["args"]
        for k in ("cache", "model", "img_size", "latent_dim", "tau", "no_density_correction", "draws_per_cycle",
                  "drop_tail_s", "drop_failed", "batch", "seed", "limit_scenarios"):
            if getattr(args, k) != saved.get(k):
                raise SystemExit(f"--{k} differs from the run in {args.out} ({getattr(args, k)} vs {saved.get(k)})")

    # data
    cache = CEMCache(args.cache)
    rows = select_cycles(cache.cycles, args.drop_tail_s, keep_failed=not args.drop_failed)
    if args.limit_scenarios:
        keep = set(cache.cycles.scenario.drop_duplicates().iloc[:args.limit_scenarios])
        rows = rows[cache.cycles.scenario.iloc[rows].isin(keep).to_numpy()]
    if args.val_cache:
        train_rows = rows
        val_cache = CEMCache(args.val_cache)
        val_rows = select_cycles(val_cache.cycles, args.drop_tail_s, keep_failed=not args.drop_failed)
    else:
        train_rows, val_rows = split_by_scenario(cache.cycles, rows, args.val_fraction, args.seed)
        val_cache = cache
    val_rows = np.random.default_rng(args.seed).permutation(val_rows)[:args.val_max_cycles]
    common = dict(tau=args.tau, density_correction=not args.no_density_correction, img_size=args.img_size)
    train_set = CEMCycleDataset(cache, train_rows, draws_per_cycle=args.draws_per_cycle, seed=args.seed, **common)
    val_set = CEMCycleDataset(val_cache, val_rows, seed=args.seed + 1, **common)

    norm_dir = os.path.join(args.out, "normalizer")
    if resume:
        normalizer = Normalizer.load(norm_dir)
    else:
        normalizer = Normalizer()
        normalizer.fit(train_set.sample_targets(min(200000, 20 * len(train_rows)), seed=args.seed))
        normalizer.save(norm_dir)
        ess = train_set.effective_sample_size()
        json.dump({**vars(args), "z_order": ["d", "v", "T"], "train_cycles": len(train_rows), "val_cycles": len(val_rows),
                   "ess_median": float(np.median(ess)), "ess_p10": float(np.percentile(ess, 10)),
                   "cache_meta": {k: v for k, v in cache.meta.items() if k != "dataset_info"}},
                  open(os.path.join(args.out, "config.json"), "w"), indent=1)
        logging.info(f"{len(train_rows)} train / {len(val_rows)} val cycles; effective candidates per cycle "
                     f"(ESS) median {np.median(ess):.1f}, 10th percentile {np.percentile(ess, 10):.1f}")
    train_set.normalizer = val_set.normalizer = normalizer

    sampler = ResumableSampler(len(train_set), seed=args.seed)
    loader_kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, shuffle=False, **loader_kw)

    # model and optimisation
    model, loss_fn = build_model(args)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.5)
    annealer = BetaAnnealer(0.0, args.beta_end, args.epochs, args.beta_schedule)
    state = {"epoch": 0, "position": 0, "sums": [0.0, 0.0, 0.0], "batches": 0, "best_val": float("inf"), "beta": None}
    if resume:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        annealer.load_state_dict(resume["annealer"])
        state = resume["state"]
        logging.info(f"resumed at epoch {state['epoch'] + 1}, item {state['position']}")

    def save_checkpoint():
        tmp = ckpt_path + ".tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "annealer": annealer.state_dict(), "state": state, "args": vars(args)}, tmp)
        os.replace(tmp, ckpt_path)  # never leaves a half-written checkpoint

    log_path = os.path.join(args.out, "log.tsv")
    if not os.path.exists(log_path):
        open(log_path, "w").write("epoch\ttrain_loss\ttrain_recon\ttrain_kl\tval_loss\tval_recon\tval_kl\tbeta\tlr\n")
    last_ckpt = time.time()

    while state["epoch"] < args.epochs:
        epoch = state["epoch"]
        if state["beta"] is None:  # first batch of the epoch
            state["beta"] = annealer.step()
        train_set.set_epoch(epoch)
        sampler.set_epoch(epoch, start=state["position"])
        loader = DataLoader(train_set, sampler=sampler, drop_last=True, **loader_kw)
        model.train()
        for x, imgs in loader:
            x, imgs = x.to(device, non_blocking=True), imgs.to(device, non_blocking=True)
            y_pred, kl_values = model(x, imgs)
            loss, rec, kl = loss_fn(y_pred, x, kl_values, beta=state["beta"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            for k, v in enumerate((loss, rec, kl)):
                state["sums"][k] += v.item()
            state["batches"] += 1
            state["position"] += len(x)
            if state["batches"] % 200 == 0:
                s = [v / state["batches"] for v in state["sums"]]
                logging.info(f"epoch {epoch + 1} item {state['position']}/{len(train_set)}: loss {s[0]:.4f} "
                             f"recon {s[1]:.4f} kl {s[2]:.4f} beta {state['beta']:.3f}")
            out_of_time = time.time() - t_start > args.max_hours * 3600
            if stop.requested or out_of_time or time.time() - last_ckpt > args.ckpt_minutes * 60:
                save_checkpoint()
                last_ckpt = time.time()
                if stop.requested or out_of_time:
                    logging.info("stopped; run the same command to resume")
                    return

        # end of epoch: validation on fixed draws, logging, best model
        model.eval()
        val = np.zeros(3)
        with torch.no_grad():
            for x, imgs in val_loader:
                y_pred, kl_values = model(x.to(device), imgs.to(device))
                val += [v.item() for v in loss_fn(y_pred, x.to(device), kl_values, beta=1.0)]
        val /= max(len(val_loader), 1)
        train = [v / max(state["batches"], 1) for v in state["sums"]]
        lr = optimizer.param_groups[0]["lr"]
        with open(log_path, "a") as fh:
            fh.write("\t".join(str(v) for v in [epoch + 1, *np.round(train, 6), *np.round(val, 6),
                                                round(state["beta"], 6), lr]) + "\n")
        logging.info(f"epoch {epoch + 1} done: train loss {train[0]:.4f}, val loss {val[0]:.4f} (recon {val[1]:.4f}, kl {val[2]:.4f})")
        if val[0] < state["best_val"]:
            state["best_val"] = float(val[0])
            torch.save(model.state_dict(), os.path.join(args.out, "model_best.pth"))
            logging.info(f"new best model (val loss {val[0]:.4f})")
        scheduler.step()
        state.update(epoch=epoch + 1, position=0, sums=[0.0, 0.0, 0.0], batches=0, beta=None)
        save_checkpoint()
        last_ckpt = time.time()
    logging.info("training complete")


if __name__ == "__main__":
    main()
