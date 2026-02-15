import os
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from cvae.model.hcvae import HierarchicalCVAE, reconstruction_kld
from cvae.model.conditional_vae import CVAE 
from cvae.model.attn_cvae import attnCVAE
from cvae.model.cnn_cvae import cnnCVAE
from cvae.model.imgs_cond_dataset import CVAEDataset
from torchvision import transforms
from normalizer import Normalizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 1
img_dim = 256
history = 3
normalize = True

num_workers = 2  # for data loading

log_every = 10

# import data
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

test_imgs_root = os.path.join(BASE_DIR, 'data/data_v2/test/imgs/')
targets_test_path = os.path.join(BASE_DIR, 'data/data_v2/test/sampled_vars.parquet')

#model_name= "hcvae"
#model_name = "cvae"
model_name = "attn_cvae"
#model_name = "cnn_cvae"
z_dim = 64

weights_path = os.path.join(BASE_DIR, 'model/weights/attn_cvae_zdim_64_sigmoid_1.0_stall_end.pth') # MSE=0.336199, Avg Recon Loss=0.3361, Avg KL=0.0041

imgs_transforms = transforms.Compose([
    #transforms.CenterCrop((514, 514)),
    #transforms.Resize((img_dim, img_dim)),
    transforms.ToTensor(),
    #transforms.Grayscale(num_output_channels=1),
    #transforms.Lambda(lambda x: 1.0 - x),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

normalizer = Normalizer()

test_dataset = CVAEDataset(
    targets_path=targets_test_path,
    image_root=test_imgs_root,
    mode="test",
    image_transform=imgs_transforms,
    normalize=normalize,  
    history=history,
    num_workers=16
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

# Load the model
if model_name == "hcvae":
    model = HierarchicalCVAE(latent_dim=z_dim, img_channels=history, img_size=img_dim, attn=True)
elif model_name == "cvae":
    model = CVAE(z_dim=z_dim)
elif model_name == "attn_cvae":
    model = attnCVAE(latent_dim=z_dim, img_channels=history, img_size=img_dim)
elif model_name == "cnn_cvae":
    model = cnnCVAE(latent_dim=z_dim, img_channels=history, img_size=img_dim)
model.load_state_dict(torch.load(weights_path, map_location=device))
model = model.to(device)
model.eval()



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
    target_mean    = [ 4.17129931, -0.03913221, 10.2505778 ]
    target_std_dev = [0.92924915, 0.30188121, 4.31719223]
    
    with torch.inference_mode():
        imgs_tensor = torch.cat(imgs_list[0:history], dim=1).to(device)
        parameters_normalized = model.generate(c=imgs_tensor, batch=num_samples, device=device)
        if normalizer is not None:
            normalizer.load_from_stats(mean=target_mean, std=target_std_dev)
            parameters = normalizer.inverse_transform_targets(parameters_normalized.cpu().numpy())
        else:
            parameters = parameters_normalized.cpu().numpy()
    
    return parameters.tolist()


# # Example usage of generate_samples function
# scenario = "ARG_Carcarana-4_1_T-1"
# scenario_dir = os.path.join(test_imgs_root, scenario)

# img_paths = [
#     os.path.join(scenario_dir, f"{i}.png")
#     for i in range(3)
# ]

# scenario_dir = 'test'
# img_paths = [
#     os.path.join(scenario_dir, f"{i}.png")
#     for i in range(3)
# ]

# imags = [Image.open(p).convert("RGB") for p in img_paths]

# parameters = generate_samples(model, imags, num_samples=5, transformation=imgs_transforms, normalizer=normalizer, device=device)
# print(f"Generated samples: {parameters}")


total_mse = 0.0
total_rec_loss = 0.0
total_kl = 0.0
count = 0

print("\nEvaluating model on test set...")

errors_dim0 = []
errors_dim1 = []
errors_dim2 = []
targets_dim0 = []
targets_dim1 = []
targets_dim2 = []
preds_un_dim0 = []
preds_un_dim1 = []
preds_un_dim2 = []
errors_norm_dim0 = []
errors_norm_dim1 = []
errors_norm_dim2 = []

with torch.inference_mode():
    test_bar = tqdm(test_loader, desc="Test MSE", ncols=100)
    for batch_targets, batch_imgs in test_bar:
        batch_imgs = batch_imgs.to(device)
        batch_targets = batch_targets.to(device)

        # Forward pass
        parameters_normalized = model.generate(
            c=batch_imgs,
            batch=batch_targets.shape[0],
            device=device
        )
        parameters_reconstructed, kl_values = model(c=batch_imgs, x=batch_targets)

        #print(parameters_normalized, batch_targets)
        mse = F.mse_loss(parameters_normalized, batch_targets)
        _, rec_loss, kl = reconstruction_kld(parameters_reconstructed, batch_targets, kl_values, beta=1.0)

        total_mse += mse.item()
        total_rec_loss += rec_loss.item()
        total_kl += kl.item()
        count += batch_targets.shape[0]

        test_bar.set_postfix({
            "MSE": f"{total_mse / count:.6f}",
            "Recon": f"{total_rec_loss / count:.4f}",
            "KL": f"{total_kl / count:.4f}"
        })

        # Unnormalize outputs and targets
        target_mean    = [ 4.17129931, -0.03913221, 10.2505778 ]
        target_std_dev = [0.92924915, 0.30188121, 4.31719223]

        normalizer.load_from_stats(mean=target_mean, std=target_std_dev)
        pred_un = normalizer.inverse_transform_targets(parameters_normalized.cpu().numpy())
        tgt_un  = normalizer.inverse_transform_targets(batch_targets.cpu().numpy())

        pred_un = torch.tensor(pred_un)
        tgt_un  = torch.tensor(tgt_un)

        # Compute per-dimension squared errors
        sq_err = (pred_un - tgt_un) ** 2

        errors_dim0.extend(sq_err[:, 0].tolist())
        errors_dim1.extend(sq_err[:, 1].tolist())
        errors_dim2.extend(sq_err[:, 2].tolist())

        # Save also the target values (for analyzing ranges)
        targets_dim0.extend(tgt_un[:, 0].tolist())
        targets_dim1.extend(tgt_un[:, 1].tolist())
        targets_dim2.extend(tgt_un[:, 2].tolist())

        preds_un_dim0.extend(pred_un[:, 0].tolist())
        preds_un_dim1.extend(pred_un[:, 1].tolist())
        preds_un_dim2.extend(pred_un[:, 2].tolist())

        sq_err_norm = (parameters_normalized - batch_targets)**2  # still on device / normalized space

        errors_norm_dim0.extend(sq_err_norm[:, 0].cpu().tolist())
        errors_norm_dim1.extend(sq_err_norm[:, 1].cpu().tolist())
        errors_norm_dim2.extend(sq_err_norm[:, 2].cpu().tolist())

mse_dim0 = sum(errors_dim0) / len(errors_dim0)
mse_dim1 = sum(errors_dim1) / len(errors_dim1)
mse_dim2 = sum(errors_dim2) / len(errors_dim2)

mse_norm_dim0 = sum(errors_norm_dim0) / len(errors_norm_dim0)
mse_norm_dim1 = sum(errors_norm_dim1) / len(errors_norm_dim1)
mse_norm_dim2 = sum(errors_norm_dim2) / len(errors_norm_dim2)

print("\nPer-dimension MSE (unnormalized):")
print(f"t0: {mse_dim0:.4f}")
print(f"t1: {mse_dim1:.4f}")
print(f"t2: {mse_dim2:.4f}")


print("\nPer-dimension MSE (normalized):")
print(f"t0: {mse_norm_dim0:.6f}")
print(f"t1: {mse_norm_dim1:.6f}")
print(f"t2: {mse_norm_dim2:.6f}")

mean_mse = total_mse / count
mean_rec_loss = total_rec_loss / count
mean_kl = total_kl / count

print(f"\n Final Test MSE={mean_mse:.6f}, Avg Recon Loss={mean_rec_loss:.4f}, Avg KL={mean_kl:.4f}")

import numpy as np
import matplotlib.pyplot as plt

t1 = np.array(targets_dim1)
e1 = np.array(errors_dim1)

bins = np.linspace(t1.min(), t1.max(), 100)
digitized = np.digitize(t1, bins)

bin_centers = []
bin_errors = []
bin_counts = []

for i in range(1, len(bins)):
    mask = digitized == i
    if mask.sum() > 0:
        mse = e1[mask].mean()
        #print(f"Range {bins[i-1]:.2f} to {bins[i]:.2f}: MSE={mse:.4f} (n={mask.sum()})")

        # store for plotting
        center = 0.5 * (bins[i-1] + bins[i])
        bin_centers.append(center)
        bin_errors.append(mse)
        bin_counts.append(mask.sum())

# ---- Ensure output folder exists ----
output_dir = "results_analysis"
os.makedirs(output_dir, exist_ok=True)

# ---- Plot and save ----
plt.figure(figsize=(9, 5))
plt.plot(bin_centers, bin_errors, marker='o')
plt.xlabel("Target value (bin centers)")
plt.ylabel("Mean Squared Error")
plt.title("Binned MSE over Target Dim 1")
plt.grid(True)
plt.tight_layout()

save_path = os.path.join(output_dir, "binned_mse_dim1.png")
plt.savefig(save_path, dpi=200)
plt.close()

print(f"Saved figure to: {save_path}")


t2 = np.array(targets_dim2)
e2 = np.array(errors_dim2)

bins = np.linspace(t2.min(), t2.max(), 100)
digitized = np.digitize(t2, bins)

bin_centers = []
bin_errors = []
bin_counts = []

for i in range(1, len(bins)):
    mask = digitized == i
    if mask.sum() > 0:
        mse = e2[mask].mean()
        #print(f"Range {bins[i-1]:.2f} to {bins[i]:.2f}: MSE={mse:.4f} (n={mask.sum()})")

        # store for plotting
        center = 0.5 * (bins[i-1] + bins[i])
        bin_centers.append(center)
        bin_errors.append(mse)
        bin_counts.append(mask.sum())

# ---- Plot and save ----
plt.figure(figsize=(9, 5))
plt.plot(bin_centers, bin_errors, marker='o')
plt.xlabel("Target value (bin centers)")
plt.ylabel("Mean Squared Error")
plt.title("Binned MSE over Target Dim 2")
plt.grid(True)
plt.tight_layout()

save_path = os.path.join(output_dir, "binned_mse_dim2.png")
plt.savefig(save_path, dpi=200)
plt.close()

print(f"Saved figure to: {save_path}")


t0 = np.array(targets_dim0)
e0 = np.array(errors_dim0)

bins = np.linspace(t0.min(), t0.max(), 100)
digitized = np.digitize(t0, bins)

bin_centers = []
bin_errors = []
bin_counts = []

for i in range(1, len(bins)):
    mask = digitized == i
    if mask.sum() > 0:
        mse = e0[mask].mean()
        #print(f"Range {bins[i-1]:.2f} to {bins[i]:.2f}: MSE={mse:.4f} (n={mask.sum()})")

        # store for plotting
        center = 0.5 * (bins[i-1] + bins[i])
        bin_centers.append(center)
        bin_errors.append(mse)
        bin_counts.append(mask.sum())

# ---- Plot and save ----
plt.figure(figsize=(9, 5))
plt.plot(bin_centers, bin_errors, marker='o')
plt.xlabel("Target value (bin centers)")
plt.ylabel("Mean Squared Error")
plt.title("Binned MSE over Target Dim 0")
plt.grid(True)
plt.tight_layout()

save_path = os.path.join(output_dir, "binned_mse_dim0.png")
plt.savefig(save_path, dpi=200)
plt.close()

print(f"Saved figure to: {save_path}")



# Convert lists to arrays
t1 = np.array(targets_dim0)
t2 = np.array(targets_dim1)
t3 = np.array(targets_dim2)

# Ensure output directory exists
output_dir = "results_analysis"
os.makedirs(output_dir, exist_ok=True)

targets = [t1, t2, t3]
names   = ["t", "d", "v"]

for i, (t, name) in enumerate(zip(targets, names), start=1):
    plt.figure(figsize=(8, 5))
    plt.hist(t, bins=80, alpha=0.75, edgecolor='black')
    plt.title(f"Histogram of {name}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    save_path = os.path.join(output_dir, f"histogram_target_{i - 1}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved: {save_path}")


# Convert to numpy for plotting
t0 = np.array(targets_dim0)
t1 = np.array(targets_dim1)
t2 = np.array(targets_dim2)

p0 = np.array(preds_un_dim0)
p1 = np.array(preds_un_dim1)
p2 = np.array(preds_un_dim2)

def save_scatter(t, p, dim_name):
    plt.figure(figsize=(6,6))
    plt.scatter(t, p, s=4, alpha=0.4)
    plt.xlabel("Target (unnormalized)")
    plt.ylabel("Prediction (unnormalized)")
    plt.title(f"Target vs Prediction — Dimension {dim_name}")
    plt.grid(True)

    # Save figure
    save_path = os.path.join(output_dir, f"scatter_dim_{dim_name}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved scatter plot for dim {dim_name} → {save_path}")

# Save all 3
save_scatter(t0, p0, "t")
save_scatter(t1, p1, "d")
save_scatter(t2, p2, "v")

def save_target_scatter(x, y, name_x, name_y):
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=4, alpha=0.4)
    plt.xlabel(f"Target {name_x} (unnormalized)")
    plt.ylabel(f"Target {name_y} (unnormalized)")
    plt.title(f"Target {name_x} vs Target {name_y}")
    plt.grid(True)
    
    save_path = os.path.join(output_dir, f"scatter_targets_{name_x}_vs_{name_y}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")

# ---- Generate all target-vs-target scatter plots ----
save_target_scatter(t0, t1, "t", "d")
save_target_scatter(t0, t2, "t", "v")
save_target_scatter(t1, t2, "d", "v")