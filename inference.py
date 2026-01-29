import os
import torch
import numpy as np
from cvae.model.hcvae import HierarchicalCVAE, reconstruction_kld
from tqdm import tqdm
import pandas as pd
from cvae.model.imgs_cond_dataset import CVAEDataset
from cvae.model.mask_background import MaskBackground
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR
import sys
import logging

logging.basicConfig(level=logging.INFO, 
                    format="[%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

batch_size = 1024
img_dim = 128
history = 3

num_workers = 16  # for data loading

log_every = 10

# import data
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

test_imgs_root = os.path.join(BASE_DIR, 'data/data_v2/test/imgs/')
targets_test_path = os.path.join(BASE_DIR, 'data/data_v2/test/sampled_vars.parquet')

weights_path = os.path.join(BASE_DIR, 'model/weights/hcvae_opt_batch_64_epochs_20_zdim_32_sigmoid_0.1_stall_end.pth')

imgs_transforms = transforms.Compose([
            transforms.Resize((img_dim, img_dim)),
            transforms.ToTensor(),  
            transforms.Grayscale(num_output_channels=1),
            transforms.Lambda(lambda x: 1.0 - x),
            transforms.Normalize( mean=[0.5], std=[0.5])
        ])

# prepare datasets and dataloaders
test_dataset = CVAEDataset(
    targets_path=targets_test_path,
    image_root=test_imgs_root,
    mode='test',
    image_transform=imgs_transforms,
    num_workers=num_workers,
    normalize=True,
    history=history
)

test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                             batch_size=batch_size,
                                             pin_memory=True,
                                             num_workers=num_workers,
                                             shuffle=True)

# Load the model
model = HierarchicalCVAE(img_channels=history, img_size=img_dim, attn=True)
model.load_state_dict(torch.load(weights_path, map_location=device))
model = model.to(device)
model.eval()


#Reconstruction + KL evaluation:
with torch.inference_mode():
    cum_loss = 0.0
    for batch in test_dataloader:
        x, imgs = batch
        x, imgs = x.to(device, non_blocking=True), imgs.to(device, non_blocking=True)
        y_pred, kl_values = model(x, imgs)
        loss = reconstruction_kld(y_pred, x, kl_values, beta=0.1)
        cum_loss += sum(loss).item()
        print(f"batch reconstruction + kl loss: {sum(loss):.4f}")
    cum_loss /= len(test_dataloader)
    print(f"Total reconstruction + kl loss: {cum_loss:.4f}")


# Generate samples:
num_trajectories_to_generate = 10
with torch.inference_mode():
    imgs_cond = test_dataset[0][1].unsqueeze(0).to(device)  # use first sample's image condition
    parameters_normalized = model.generate(c=imgs_cond, batch=num_trajectories_to_generate, device=device)
    parameters = test_dataset.normalizer.inverse_transform_targets(parameters_normalized.cpu().numpy())
    print(f"Generated sample: {parameters}")