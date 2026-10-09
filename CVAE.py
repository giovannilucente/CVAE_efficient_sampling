import os
import numpy as np
import torch
from PIL import Image
from .hcvae import HierarchicalCVAE
from .imgs_cond_dataset import CVAEDataset
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


class CostAwareCVAE:
    """Sampler of a model trained by train_cost_cvae.py: BEV history of the current planning cycle
    -> sampling parameters z = [d, v, T] (terminal lateral offset, terminal speed, horizon).

    run_dir holds config.json, normalizer/ and the weights. Images are converted exactly as in
    training (cem_dataset.bev_tensor).
    """

    HISTORY = 3

    def __init__(self, run_dir: str, device: torch.device, weights: str = "model_best.pth"):
        import json
        from .cem_dataset import bev_tensor
        self._bev_tensor = bev_tensor
        self.device = device
        self.config = json.load(open(os.path.join(run_dir, "config.json")))
        self.img_size = self.config["img_size"]
        if self.config["model"] == "attn":
            self.model = attnCVAE(hidden_dim=32, input_dim=3, img_channels=self.HISTORY,
                                  img_size=self.img_size, latent_dim=self.config["latent_dim"])
        else:
            self.model = HierarchicalCVAE(hidden_dim=32, input_dim=3, img_channels=self.HISTORY,
                                          img_size=self.img_size, latent_dim=self.config["latent_dim"], attn=True)
        self.model.load_state_dict(torch.load(os.path.join(run_dir, weights), map_location=device))
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
    def generate_samples(self, frames, num_samples: int) -> np.ndarray:
        """num_samples x 3 array of z = [d, v, T]."""
        z = self.model.generate(c=self.condition(frames), batch=num_samples, device=self.device)
        return self.normalizer.inverse_transform_targets(z.cpu().numpy())
