# Efficient sampling through a Hierarchical Conditional Variational Autoencoder (HCVAE)

The weights are in the weight folder. 
You can check how to generate samples with the trained model in the ```inference.py``` script, by using the function ``` generate_samples(model, imgs_list, num_samples, transformation, normalizer, device)``` :
```bash

def generate_samples(model, imgs_list, num_samples, transformation=None, normalizer=None, device=None):
    history = 3
    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if transformation is not None:
        imgs_list = [transformation(img).unsqueeze(0) for img in imgs_list] 
    else:
        print("The model expects transformed images as input.")
        return None

    # Dataset statistics [t, d, v]    
    target_mean    = [ 4.69328707, -0.03879964, 10.74773858]
    target_std_dev = [0.67665561, 0.23729723, 3.20131289]
    
    with torch.inference_mode():
        imgs_tensor = torch.cat(imgs_list[0:history], dim=1).to(device)
        parameters_normalized = model.generate(c=imgs_tensor, batch=num_samples, device=device)
        if normalizer is not None:
            normalizer.load_from_stats(mean=target_mean, std=target_std_dev)
            parameters = normalizer.inverse_transform_targets(parameters_normalized.cpu().numpy())
        else:
            parameters = parameters_normalized.cpu().numpy()
    
    return parameters.tolist()


# Example usage of generate_samples function
scenario = "ARG_Carcarana-4_1_T-1"
scenario_dir = os.path.join(test_imgs_root, scenario)

img_paths = [
    os.path.join(scenario_dir, f"{i}.png")
    for i in range(3)
]

imags = [Image.open(p).convert("RGB") for p in img_paths]
normalizer = Normalizer()
parameters = generate_samples(model, imags, num_samples=5, transformation=imgs_transforms, normalizer=normalizer, device=device)
print(f"Generated samples: {parameters}")
```
Remember to load the normalizer and the transformations needed for the images.
For this model I used a history of 3 BEV images with dimension ``` img_dim=128``` , in black and white, saved in a 3 channel tensor. The images of the dataset need to have these transformations before getting passed to the model:
```bash
imgs_transforms = transforms.Compose([
            transforms.Resize((img_dim, img_dim)),
            transforms.ToTensor(),  
            transforms.Grayscale(num_output_channels=1),
            transforms.Lambda(lambda x: 1.0 - x),
            transforms.Normalize( mean=[0.5], std=[0.5])
        ])
```
So the images are resized with the dimension 128x128 pixels, then transformed in tensors, then transformed in a grayscale, then through the function Lambda inverted (black background and traffic participants white). Finally the images are normalized.


# Cost-aware training on CEM data (branch cost-aware)

The training data come from fiss_plus_planner: `Collect_Data_For_ML` with `PLANNER: CEM_CPP`
stores every CEM candidate of every planning cycle (see the main repository README). For the
scene of a planning cycle (3 BEV frames), the CVAE learns the distribution of the planner's
sampling parameters z = [d, v, T]: terminal lateral offset, terminal speed and horizon, in this
order everywhere.

**Cost-aware.** Every training item is one planning cycle with one of its feasible CEM
candidates. The candidate is drawn with probability proportional to exp(-J~ / tau) / q(z):
- J~ is the candidate's cost, normalised within the cycle (0 = best, 1 = 90th percentile);
- q is CEM's sampling density. Dividing by q removes CEM's concentration of samples near its
  optimum, so only the cost decides how often a candidate is learned.

Drawing the targets this way and training with the usual ELBO is, in expectation, the
cost-weighted ELBO. The CVAE therefore learns p(z | scene) proportional to exp(-J~ / tau), and
the model code itself is unchanged.

## Files

| file | purpose |
|---|---|
| `cem_cache.py` | builds the training cache from a collection (once per data set) |
| `cem_dataset.py` | dataset on the cache: cost-aware target draws, cycle selection, scenario split |
| `train_cost_cvae.py` | training; checkpoints, stops cleanly and resumes (HPC time limits) |
| `test_cost_cvae.py` | checks a trained model through the inference interface on held-out cycles |
| `CVAE.py: CostAwareCVAE` | inference interface: BEV frames -> samples [d, v, T] |
| `slurm_train.sh` | example Slurm job |
| `beta_annealer.py` | KL weight schedule |

## 1. Build the cache (on the machine that holds the collected data)

```bash
cd CVAE_efficient_sampling
python cem_cache.py --data <collection OUTPUT_DIR> --out <cache dir> --workers 16
cd <cache dir> && sha256sum *.bin cycles.parquet meta.json > sha256sums.txt
```

The cache holds flat memory-mapped arrays (about 47 GB for the 6683 Train scenarios, about 1 min
to build):
- per planning cycle: the scenario outcome, the search space and the 3-frame image history;
- per feasible candidate: z, J~ and log q.

Scenarios whose files cannot be read are skipped and listed in `meta.json` under `skipped`. Only
the cache is needed for training, not the raw collection.

## 2. Set up the HPC

```bash
git clone --recurse-submodules <fiss_plus_planner repo> && cd fiss_plus_planner
git -C CVAE_efficient_sampling checkout cost-aware && git -C CVAE_efficient_sampling pull
conda create -n cvae python=3.10 && conda activate cvae
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.x build
pip install numpy pandas pyarrow scikit-learn pillow matplotlib tqdm
# pretrained ResNet18 of attnCVAE: download once on a login node (compute nodes may have no internet)
python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)"
# copy the cache and verify it
rsync -a --progress <local cache dir>/ $SCRATCH/cem_train_cache/
cd $SCRATCH/cem_train_cache && sha256sum -c sha256sums.txt
```

## 3. Smoke test (about 5 min on one GPU; can run interactively)

```bash
cd fiss_plus_planner/CVAE_efficient_sampling
python train_cost_cvae.py --cache $SCRATCH/cem_train_cache --out runs/smoke \
    --limit_scenarios 300 --epochs 2 --draws_per_cycle 4
python test_cost_cvae.py --run runs/smoke --cycles 300 --k 16
```

`test_cost_cvae.py` compares, on held-out cycles, the CVAE samples with two samplers that ignore
the scene: uniform in the cycle's search space, and the overall target distribution
("marginal"):

| column | meaning |
|---|---|
| `in_bounds` | share of samples inside the cycle's search space |
| `best_dist` | distance (std units) of the closest sample to CEM's best candidate |
| `best_cost@K` | J~ of the feasible candidate nearest to the best sample (0 = CEM's best); a proxy of the cost the planner would reach with K samples |
| `mean_dist` | distance of the samples' mean to the mean of the cycle's target distribution: does the CVAE use the scene? |
| `spread` | std of the samples (std units); compare with the printed target spread |

Smoke test on a local RTX 5070 Ti (300 scenarios, 2 epochs):

| sampler | in_bounds | best_dist | best_cost@K | mean_dist | spread |
|---|---|---|---|---|---|
| cvae | 0.907 | 1.155 | 0.547 | **0.707** | 0.029 |
| uniform | 1.000 | 1.171 | 0.287 | 1.508 | 1.267 |
| marginal | 0.699 | 0.838 | 0.108 | 1.270 | 0.957 |

The target spread was 0.607. `mean_dist` shows that the pipeline works: the CVAE follows the
scene. Its `spread` is far below the target spread, though: with the KL weight rising to
`--beta_end 1.0`, the short run collapsed to almost one point per scene. This is posterior
collapse, so best-of-K is not better than the marginal sampler yet. Watch `spread` in the full
runs; options are a lower `--beta_end` (e.g. 0.1, as the original HCVAE training) or `--model
hcvae`.

## 4. Full training

```bash
sbatch slurm_train.sh                                         # edit CACHE / RUN / partition first
sbatch slurm_train.sh --tau 0.25 --beta_end 0.1               # extra arguments go to the training
```

**Checkpoints and resuming.**
- A checkpoint is written every `--ckpt_minutes` (30) and at the end of every epoch.
- Slurm sends SIGUSR1 10 min before the time limit; the run then saves and stops cleanly.
  `--max_hours` is a backup limit.
- To continue, submit the same command again. It resumes at the same position of the epoch.
  Changing the data or sampling options of an existing run directory is refused.

**Outputs in the run directory:**
- `model_best.pth` (best validation loss);
- `normalizer/` (mean / std of [d, v, T]);
- `config.json` (all settings, cycle counts, effective candidates per cycle);
- `log.tsv` (per epoch: train / val loss, recon, KL, beta, lr), `train.log` and `ckpt_last.pt`.

**Measured locally** (attnCVAE, batch 64): about 1200 items/s, 7 min per epoch over the 480k Train
cycles, 2 GB GPU memory. An A40 or A100 has far more than needed.

**Main options:**
- `--model attn|hcvae` (attnCVAE is the model the planners use);
- `--tau` (cost temperature);
- `--no_density_correction`;
- `--draws_per_cycle`;
- `--drop_tail_s` / `--drop_failed` (cycles of failed scenarios: by default the last 2.5 s before
  a non-rear-end failure are dropped);
- `--val_fraction` or `--val_cache`;
- `--beta_end`, `--epochs`, `--lr`, `--batch`.

## 5. Inference interface

```python
from CVAE_efficient_sampling.CVAE import CostAwareCVAE
sampler = CostAwareCVAE("runs/<run>", device=torch.device("cuda"))
z = sampler.generate_samples(frames, num_samples=32)   # numpy (32, 3): [d, v, T]
```

`frames` are the last up to 3 BEV images of the planning cycle, oldest first (PIL images from
`ScenarioDrawer`, or grayscale arrays). They are converted exactly as in training. A shorter
history at the start of a scenario is padded with the first frame (t = 1 -> frames [0, 0, 1]).
The image is encoded once for all samples. The planner should clip the samples to its search
space.

The old `CVAE_Efficient` class is unchanged. It serves the released weights, with the [t, d, v]
order and fixed statistics.
