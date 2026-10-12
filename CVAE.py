import hashlib
import io
import os
import numpy as np
import torch
from PIL import Image
from .hcvae import HierarchicalCVAE
from .attn_cvae import attnCVAE
from torchvision import transforms
from .normalizer import Normalizer

class CVAE_Efficient():
    def __init__(self, device: torch.device, model_path: str):
        self.device = device
        self.model_path = model_path

        self.batch_size = 1024
        self.img_dim = 256
        self.frame_size = 3
        self.latent_dim = 64
        self.num_workers = 16  # for data loading
        
        self.model = self.initialize_model(weights_path=model_path, frame_size=self.frame_size, img_dim=self.img_dim, device=device)
        self.imgs_transforms = self.initialize_transforms(img_dim=self.img_dim)

    def initialize_model(self, weights_path: str, frame_size: int, img_dim: int, device:torch.device)->attnCVAE:
        model = attnCVAE(latent_dim=self.latent_dim, img_channels=self.frame_size, img_size=self.img_dim)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model

    def initialize_transforms(self, img_dim:int)->transforms.Compose:
        imgs_transforms = transforms.Compose([
                    # transforms.CenterCrop((800, 800)),
                    # transforms.Resize((img_dim, img_dim)),
                    transforms.ToTensor(),  
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Lambda(lambda x: 1.0 - x),
                    transforms.Normalize( mean=[0.5], std=[0.5])
                ])
        return imgs_transforms

    def generate_samples(self, imgs_list, num_samples):
        self.model.eval()
        normalizer = Normalizer()

        imgs_list = [self.imgs_transforms(img).unsqueeze(0) for img in imgs_list] 

        # Dataset statistics [t, d, v]    
        target_mean    = [ 4.3677835e+00, -2.3065007e-03,  1.0874050e+01]
        target_std_dev = [0.8905386,  0.13080278, 4.5282516 ]
        
        with torch.inference_mode():
            imgs_tensor = torch.cat(imgs_list[0:self.frame_size], dim=1).to(self.device)
            parameters_normalized = self.model.generate(c=imgs_tensor, batch=num_samples, device=self.device)
            # TODO: Giovanni: Do we really need check this?
            if normalizer is not None:
                normalizer.load_from_stats(mean=target_mean, std=target_std_dev)
                parameters = normalizer.inverse_transform_targets(parameters_normalized.cpu().numpy())
            else:
                parameters = parameters_normalized.cpu().numpy()
        
        return parameters.tolist()


def read_verified(path: str) -> bytes:
    """Bytes of a weights file, checked against SHA256SUMS in its folder when it lists the file. On a
    mismatch the file's cached pages are dropped and it is read again: the local PC has a RAM fault that
    corrupts the page cache (a single flipped bit in the v1 weights changed closed-loop results)."""
    sums, name, want = os.path.join(os.path.dirname(path), "SHA256SUMS"), os.path.basename(path), None
    if os.path.exists(sums):
        want = next((h for h, f in (line.split(maxsplit=1) for line in open(sums)) if f.strip() == name), None)
    for _ in range(2):
        data = open(path, "rb").read()
        if want is None or hashlib.sha256(data).hexdigest() == want:
            return data
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    raise RuntimeError(f"{path}: sha256 differs from {sums} also after reading it again from disk")


class CostAwareCVAE:
    """Sampler of a model trained by train_cost_cvae.py: BEV history of the current planning cycle
    -> sampling parameters z = [d, v, T] (terminal lateral offset, terminal speed, horizon).

    run_dir holds config.json, normalizer/ and the weights. Images are converted exactly as in
    training (cem_dataset.bev_tensor). A model trained with --ego also needs the cycle's full raw ego
    state (cem_dataset.EGO_COLUMNS order: v, a, s_d, s_dd, d, d_d, d_dd, v_des); the model's own columns
    (config ego_columns, default all) are taken from it and scaled as in training.
    """

    HISTORY = 3

    def __init__(self, run_dir: str, device: torch.device, weights: str = "model_best.pth"):
        import json
        from .cem_dataset import EgoScaler, bev_tensor
        self._bev_tensor = bev_tensor
        self.device = device
        self.config = json.load(open(os.path.join(run_dir, "config.json")))
        self.img_size = self.config["img_size"]
        self.uses_ego = bool(self.config.get("ego", False))
        self.ego_scaler = EgoScaler.load(os.path.join(run_dir, "normalizer", "ego_scaler.json")) if self.uses_ego else None
        if self.config["model"] == "attn":
            self.model = attnCVAE(hidden_dim=32, input_dim=3, img_channels=self.HISTORY, img_size=self.img_size,
                                  latent_dim=self.config["latent_dim"],
                                  ego_dim=len(self.ego_scaler.columns) if self.uses_ego else 0)
        else:
            self.model = HierarchicalCVAE(hidden_dim=32, input_dim=3, img_channels=self.HISTORY,
                                          img_size=self.img_size, latent_dim=self.config["latent_dim"], attn=True)
        self.model.load_state_dict(torch.load(io.BytesIO(read_verified(os.path.join(run_dir, weights))),
                                              map_location=device))
        self.model.to(device).eval()
        self.normalizer = Normalizer.load(os.path.join(run_dir, "normalizer"))

    def condition(self, frames) -> torch.Tensor:
        """frames: the last up to 3 BEV images (PIL or arrays), oldest first. At the start of a
        scenario the history is padded with the first frame, as in training (t = 1 -> [0, 0, 1])."""
        frames = list(frames)[-self.HISTORY:]
        frames = [frames[0]] * (self.HISTORY - len(frames)) + frames
        gray = [np.asarray(f.convert("L")) if isinstance(f, Image.Image) else np.asarray(f) for f in frames]
        return self._bev_tensor(gray, self.img_size)[None].to(self.device)

    @torch.no_grad()
    def generate_samples(self, frames, num_samples: int, ego=None) -> np.ndarray:
        """num_samples x 3 array of z = [d, v, T]; ego: raw ego state, needed by models trained with --ego."""
        kw = {}
        if self.uses_ego:
            if ego is None:
                raise ValueError("this CVAE is conditioned on the ego state: pass ego")
            kw["e"] = torch.from_numpy(self.ego_scaler.transform(self.ego_scaler.select(ego)))[None]
        z = self.model.generate(c=self.condition(frames), batch=num_samples, device=self.device, **kw)
        return self.normalizer.inverse_transform_targets(z.cpu().numpy())
