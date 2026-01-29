import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
import logging
import pandas as pd
import matplotlib.pyplot as plt

class Normalizer:
    def __init__(self):
        self.target_scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, targets_data):
        logging.info("Fitting normalizers on training data...")
        self.target_scaler.fit(targets_data)
        self.is_fitted = True
        
        # Log statistics
        logging.info(f"Target features - Mean: {self.target_scaler.mean_}, Std: {self.target_scaler.scale_}")
        
    def transform_targets(self, targets):
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform")
        return self.target_scaler.transform(targets).astype(np.float32)
    
    def inverse_transform_targets(self, normalized_targets):
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform")
        
        if hasattr(normalized_targets, 'cpu'):
            # PyTorch tensor
            normalized_targets = normalized_targets.cpu().numpy()
        
        return self.target_scaler.inverse_transform(normalized_targets)
    
    def save(self, save_dir):
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted normalizer")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save scalers
        target_scaler_path = os.path.join(save_dir, 'target_scaler.pkl')
        
        with open(target_scaler_path, 'wb') as f:
            pickle.dump(self.target_scaler, f)
        
        # Also save as readable text for verification
        stats_path = os.path.join(save_dir, 'normalization_stats.txt')
        with open(stats_path, 'w') as f:
            f.write("TARGET SCALER STATISTICS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Mean: {self.target_scaler.mean_}\n")
            f.write(f"Std: {self.target_scaler.scale_}\n\n")
        
        logging.info(f"Normalizer saved to {save_dir}")
    
    def load_from_stats(self, mean, std):
        self.target_scaler.mean_ = np.array(mean, dtype=np.float64)
        self.target_scaler.scale_ = np.array(std, dtype=np.float64)
        self.target_scaler.var_ = self.target_scaler.scale_ ** 2
        self.is_fitted = True
    
    @classmethod
    def load(cls, save_dir):
        normalizer = cls()
        
        target_scaler_path = os.path.join(save_dir, 'target_scaler.pkl')
        
        if not os.path.exists(target_scaler_path):
            raise FileNotFoundError(f"Scaler files not found in {save_dir}")
        
        with open(target_scaler_path, 'rb') as f:
            normalizer.target_scaler = pickle.load(f)
        
        normalizer.is_fitted = True
        
        logging.info(f"Normalizer loaded from {save_dir}")
        logging.info(f"Target mean: {normalizer.target_scaler.mean_}")
        logging.info(f"Target std: {normalizer.target_scaler.scale_}")
        
        return normalizer

def plot_histograms(df, features, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for feat in features:
        plt.figure()
        plt.hist(df[feat].dropna(), bins=60)
        plt.title(f"Histogram of {feat}")
        plt.xlabel(feat)
        plt.ylabel("Frequency")
        plt.tight_layout()

        out_path = os.path.join(save_dir, f"hist_{feat}.png")
        plt.savefig(out_path)
        plt.close()

        logging.info(f"Saved histogram for {feat} → {out_path}")

def main(targets_path, save_dir):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    logging.info(f"Loading dataset from {targets_path}...")
    df = pd.read_parquet(targets_path)

    target_features = ["t", "d", "v"]
    targets_np = df[target_features].to_numpy(dtype=np.float32)

    # Compute Global Stats
    mean = targets_np.mean(axis=0)
    std = targets_np.std(axis=0)
    min_vals = targets_np.min(axis=0)
    max_vals = targets_np.max(axis=0)

    logging.info("Global Target Statistics")
    logging.info("=" * 60)
    logging.info(f"Mean     : {mean}")
    logging.info(f"Std      : {std}")
    logging.info(f"Min      : {min_vals}")
    logging.info(f"Max      : {max_vals}")
    logging.info("=" * 60)

    # Fit and Save Normalizer
    normalizer = Normalizer()
    normalizer.fit(targets_np)
    normalizer.save(save_dir)

    plot_histograms(df, ["t", "d", "v"], save_dir)

    logging.info("Done.")


if __name__ == "__main__":

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    targets_train_path = os.path.join(BASE_DIR, 'data/data_v2/train/sampled_vars.parquet')
    
    main(
        targets_path=targets_train_path,
        save_dir="statistics/"
    )

    