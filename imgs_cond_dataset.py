import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import os
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from normalizer import Normalizer

logging.basicConfig(level=logging.INFO, 
                    format="[%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])


class CVAEDataset(Dataset):
    def __init__(self, 
                 targets_path,
                 image_root,
                 mode,
                 image_transform=None,
                 normalize=True,
                 num_workers=8,
                 history=5):   # number of previous frames to return

        self.mode = mode
        self.normalizer_dir = 'cvae/model/weights/'
        self.history = history
        self.image_root = image_root
        self.image_transform = image_transform

        # --- Load tabular data ---
        targets_df = pd.read_parquet(targets_path)

        # --- Targets only ---
        self.target_features = ["t", "d", "v"]
        self.targets_np = targets_df[self.target_features].to_numpy(dtype="float32")

        if normalize:
            self._setup_normalizer()

        # --- Image metadata ---
        self.scenarios = targets_df["scenario"].tolist()
        self.time_steps = targets_df["time_step"].tolist()

        # Build sequential mapping per index
        self.index_info = list(zip(self.scenarios, self.time_steps))

        # Preload all unique images
        unique_images = list(set(self.index_info))
        logging.info(f"Preloading {len(unique_images)} unique images using {num_workers} workers...")

        self.image_cache = {}
        self._preload_images_parallel(unique_images, num_workers)
        logging.info(f"Preloading complete. Cache contains {len(self.image_cache)} images.")


    # -----------------------
    # Image loading helpers
    # -----------------------
    def _load_single_image(self, scenario, time_step):
        try:
            img_path = os.path.join(self.image_root, scenario, f"{time_step}.png")
            img = Image.open(img_path).convert("L")
            if self.image_transform:
                img = self.image_transform(img)
            return (scenario, time_step), img
        except Exception as e:
            logging.error(f"Error loading {scenario}/{time_step}.png: {e}")
            return (scenario, time_step), None

    def _preload_images_parallel(self, unique_images, num_workers):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self._load_single_image, s, t): (s, t)
                for s, t in unique_images
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading images"):
                key, img = future.result()
                if img is not None:
                    self.image_cache[key] = img


    # -----------------------
    # Normalizer
    # -----------------------
    def _setup_normalizer(self):
        if self.mode == 'train':
            logging.info("[TRAIN] Fitting normalizer on training data...")
            self.normalizer = Normalizer()
            self.normalizer.fit(self.targets_np)
            os.makedirs(self.normalizer_dir, exist_ok=True)
            self.normalizer.save(self.normalizer_dir)

            self.targets_np = self.normalizer.transform_targets(self.targets_np)
            logging.info("[TRAIN] Data normalized")

        elif self.mode in ['val', 'test']:
            logging.info(f"[{self.mode.upper()}] Loading normalizer...")
            self.normalizer = Normalizer.load(self.normalizer_dir)
            if not self.normalizer.is_fitted:
                raise RuntimeError("Normalizer is not fitted. Train first.")

            self.targets_np = self.normalizer.transform_targets(self.targets_np)
            logging.info(f"[{self.mode.upper()}] Data normalized")

        else:
            raise ValueError(f"Invalid mode: {self.mode}")


    # -----------------------
    # Dataset API
    # -----------------------
    def __len__(self):
        return len(self.targets_np)

    def _get_single_image(self, scenario, time_step):
        return self.image_cache[(scenario, time_step)]

    def _get_image_seq(self, idx):
        """
        Returns a single tensor shaped (C*T, H, W):
        Concatenates history frames along the channel dimension.
        """
        scenario, time_step = self.index_info[idx]
        seq = []

        for h in range(self.history):
            step = time_step - (self.history - 1 - h)

            # Clamp to earliest frame of scenario
            if step < 0:
                step = 0

            key = (scenario, step)
            if key in self.image_cache:
                img = self.image_cache[key]
            else:
                fallback_step = max(0, min(step, time_step))
                img = self.image_cache[(scenario, fallback_step)]

            seq.append(img)

        return torch.cat(seq, dim=0)     # (C*T, H, W)


    def __getitem__(self, idx):
        target = torch.from_numpy(self.targets_np[idx])
        img_seq = self._get_image_seq(idx)
        return target, img_seq

if __name__ == "__main__":

    from torchvision import transforms
    
    img_dim = 128

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    targets_train_path = os.path.join(BASE_DIR, 'data/data_v2/train/sampled_vars.parquet')

    imgs_transforms = transforms.Compose([
        transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor(),  
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(lambda x: 1.0 - x),
        transforms.Normalize( mean=[0.5], std=[0.5])
    ])
    
    dataset = CVAEDataset(
        targets_path=targets_train_path,
        image_root=os.path.join(BASE_DIR, 'data/data_v2/train/imgs/'),
        mode='train',
        image_transform=imgs_transforms,
        normalize=True,
        num_workers=4,
        history=3
    )

    print("\nExamples of normalized targets:")
    for i in range(5):
        print(f"Sample {i}: {dataset.targets_np[i]}")
        print(f"Denormalized: {dataset.normalizer.inverse_transform_targets(dataset.targets_np[i].reshape(1, -1))}")


    