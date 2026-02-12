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

#weights_path = os.path.join(BASE_DIR, 'model/weights/cvae_zdim_64_sigmoid_1.0_stall_end.pth') # MSE=0.373193, Avg Recon Loss=0.3728, Avg KL=0.0002
#weights_path = os.path.join(BASE_DIR, 'model/weights/cnn_cvae_zdim_64_sigmoid_1.0_stall_end.pth') # MSE=0.465423, Avg Recon Loss=0.4093, Avg KL=0.0484
weights_path = os.path.join(BASE_DIR, 'model/weights/attn_cvae_zdim_64_sigmoid_1.0_stall_end.pth') # MSE=0.336199, Avg Recon Loss=0.3361, Avg KL=0.0041

imgs_transforms = transforms.Compose([
    transforms.CenterCrop((514, 514)),
    transforms.Resize((img_dim, img_dim)),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(lambda x: 1.0 - x),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

normalizer = Normalizer()

test_dataset = CVAEDataset(
    targets_path=targets_test_path,
    image_root=test_imgs_root,
    mode="test",
    image_transform=imgs_transforms,
    normalize=True,  
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


# Example usage of generate_samples function
scenario = "ARG_Carcarana-4_1_T-1"
scenario_dir = os.path.join(test_imgs_root, scenario)

img_paths = [
    os.path.join(scenario_dir, f"{i}.png")
    for i in range(3)
]

scenario_dir = 'test'
img_paths = [
    os.path.join(scenario_dir, f"{i}.png")
    for i in range(3)
]

imags = [Image.open(p).convert("RGB") for p in img_paths]

parameters = generate_samples(model, imags, num_samples=5, transformation=imgs_transforms, normalizer=normalizer, device=device)
print(f"Generated samples: {parameters}")


# total_mse = 0.0
# total_rec_loss = 0.0
# total_kl = 0.0
# count = 0

# print("\nEvaluating model on test set...")

# with torch.inference_mode():
#     test_bar = tqdm(test_loader, desc="Test MSE", ncols=100)
#     for batch_targets, batch_imgs in test_bar:
#         batch_imgs = batch_imgs.to(device)
#         batch_targets = batch_targets.to(device)

#         # Forward pass
#         parameters_normalized = model.generate(
#             c=batch_imgs,
#             batch=batch_targets.shape[0],
#             device=device
#         )
#         parameters_reconstructed, kl_values = model(c=batch_imgs, x=batch_targets)

#         #print(parameters_normalized, batch_targets)
#         mse = F.mse_loss(parameters_normalized, batch_targets)
#         _, rec_loss, kl = reconstruction_kld(parameters_reconstructed, batch_targets, kl_values, beta=1.0)

#         total_mse += mse.item()
#         total_rec_loss += rec_loss.item()
#         total_kl += kl.item()
#         count += batch_targets.shape[0]

#         test_bar.set_postfix({
#             "MSE": f"{total_mse / count:.6f}",
#             "Recon": f"{total_rec_loss / count:.4f}",
#             "KL": f"{total_kl / count:.4f}"
#         })

# mean_mse = total_mse / count
# mean_rec_loss = total_rec_loss / count
# mean_kl = total_kl / count

# print(f"\n Final Test MSE={mean_mse:.6f}, Avg Recon Loss={mean_rec_loss:.4f}, Avg KL={mean_kl:.4f}")