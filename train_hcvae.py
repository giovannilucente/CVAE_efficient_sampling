import os
import torch
import numpy as np
from hcvae import HierarchicalCVAE, reconstruction_kld
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from cvae.utils.beta_annealer import BetaAnnealer, CyclicalAnnealer
from imgs_cond_dataset import CVAEDataset
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR
import sys
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, 
                    format="[%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


batch_size = 64
img_features = 64
img_dim = 256
z_dim = 32
x_dim = 3
c_dim = 6 + img_features  # states + img features
h_Q_dim = 512
h_P_dim = 512

num_workers = 4  # for data loading

log_every = 10  # batches

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

train_imgs_root = os.path.join(BASE_DIR, 'data/data_v2/train/imgs/')
targets_train_path = os.path.join(BASE_DIR, 'data/data_v2/train/sampled_vars.parquet')

val_imgs_root = os.path.join(BASE_DIR, 'data/data_v2/val/imgs/')
targets_val_path = os.path.join(BASE_DIR, 'data/data_v2/val/sampled_vars.parquet')

imgs_transforms = transforms.Compose([
    transforms.Resize((img_dim, img_dim)),
    transforms.ToTensor(),  
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(lambda x: 1.0 - x),
    transforms.Normalize( mean=[0.5], std=[0.5])
])

# prepare datasets and dataloaders
train_dataset = CVAEDataset(
    targets_path=targets_train_path,
    image_root=train_imgs_root,
    mode='train',
    image_transform=imgs_transforms,
    normalize=True,
    num_workers=num_workers,
    history=3
)

val_dataset = CVAEDataset(
    targets_path=targets_val_path,
    image_root=val_imgs_root,
    mode='val',
    image_transform=imgs_transforms,
    normalize=True,
    num_workers=num_workers,
    history=3
)

train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size,
                                               pin_memory=True,
                                               num_workers=num_workers,
                                               shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=batch_size,
                                             pin_memory=True,
                                             num_workers=num_workers,
                                             shuffle=False)

logging.info("Data Ready!")

lr = 1e-4
num_epochs = 20
stall_epochs = 0
model_name = "hcvae_opt_256"
history = 3

kl_beta = 0.0  # KL divergence weight
beta_end = 0.1  # final KL divergence weight


kl_beta_annealer = BetaAnnealer(
    beta_start=kl_beta,
    beta_end=beta_end,
    n_steps=num_epochs,
    schedule='sigmoid'
)

# model
model = HierarchicalCVAE(img_channels=history, img_size=img_dim, attn=True).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                    milestones=[5],
                                                    gamma=0.5)

txt_log_dir = os.path.join(
    BASE_DIR,
    f"runs/model_{model_name}_lr_{lr}_batch_{batch_size}_epochs_{num_epochs}_{kl_beta_annealer.schedule}_{beta_end}"
)

os.makedirs(txt_log_dir, exist_ok=True)
txt_log_path = os.path.join(txt_log_dir, "training_log.txt")

def log_to_txt(epoch, metrics: dict):
    if epoch == 0 and not os.path.exists(txt_log_path):
        with open(txt_log_path, "w") as f:
            header = "Epoch\t" + "\t".join(metrics.keys()) + "\n"
            f.write(header)

    with open(txt_log_path, "a") as f:
        line = f"{epoch+1}\t" + "\t".join(f"{v:.6f}" for v in metrics.values()) + "\n"
        f.write(line)


best_val_loss = float('inf')

# Training loop
for epoch in range(num_epochs):
    train_loss, train_loss_beta_1, recon_loss, kl_loss = 0.0, 0.0, 0.0, 0.0
    kl_loss_beta_1 = 0.0
    
    # ----- Training Step -----
    model.train()
    train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    if epoch >= stall_epochs:
        kl_beta = kl_beta_annealer.step()

    for i, batch in enumerate(train_bar):
        # start_data.record()
        x, imgs = batch
        x, imgs = x.to(device, non_blocking=True), imgs.to(device, non_blocking=True)
        # end_data.record()
        
        # start_forward.record()
        y_pred, kl_values = model(x, imgs)
        loss, rec_loss, kl = reconstruction_kld(y_pred, x, kl_values, beta=kl_beta)
        # end_forward.record()
        
        # start_backward.record()
        optimizer.zero_grad(set_to_none=True)  # clear gradients more effieciently (faster and saves memory)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # scheduler.step()
        # end_backward.record()
        
        recon_loss += rec_loss.item()
        kl_loss += kl.item()
        train_loss += loss.item()
        train_loss_beta_1 += reconstruction_kld(y_pred, x, kl_values, beta=1.0)[0].item()
        kl_loss_beta_1 += reconstruction_kld(y_pred, x, kl_values, beta=1.0)[2].item()

        avg_recon_loss = recon_loss / (i + 1)
        avg_kl_loss = kl_loss / (i + 1)
        avg_train_loss = train_loss / (i + 1)

        # update tqdm bar
        train_bar.set_postfix({
            "Recon": f"{avg_recon_loss:.4f}",
            "KL": f"{avg_kl_loss:.4f}",
            "Total": f"{avg_train_loss:.4f}",
            "KL_beta": f"{kl_beta:.4f}"
        })
            
    avg_train_recon_loss = recon_loss / len(train_dataloader)
    avg_train_kl_loss = kl_loss / len(train_dataloader)
    avg_train_loss = train_loss / len(train_dataloader)
    
    avg_train_loss_beta_1 = train_loss_beta_1 / len(train_dataloader)
    avg_train_kl_loss_beta_1 = kl_loss_beta_1 / len(train_dataloader)
        
    # ----- Validation Step -----
    model.eval()
    val_loss, recon_loss, kl_loss = 0.0, 0.0, 0.0
    val_bar = tqdm(val_dataloader, desc=f"Val {epoch+1}/{num_epochs}", leave=False)
    with torch.no_grad():
        for i, batch in enumerate(val_bar):
            x, imgs = batch
            x, imgs = x.to(device, non_blocking=True), imgs.to(device, non_blocking=True)
            
            y_pred, kl_values = model(x, imgs)
            loss, rec_loss, kl = reconstruction_kld(y_pred, x, kl_values, beta=1.0)

            recon_loss += rec_loss.item()
            kl_loss += kl.item()
            val_loss += loss.item()

            avg_recon_loss = recon_loss / (i + 1)
            avg_kl_loss = kl_loss / (i + 1)
            avg_val_loss = val_loss / (i + 1)

            val_bar.set_postfix({
                "Recon": f"{avg_recon_loss:.4f}",
                "KL": f"{avg_kl_loss:.4f}",
                "Total": f"{avg_val_loss:.4f}"
            })
            
    avg_val_recon_loss = recon_loss / len(val_dataloader)
    avg_val_kl_loss = kl_loss / len(val_dataloader)
    avg_val_loss = val_loss / len(val_dataloader)
    
    lr_scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # ----- TXT LOGGING -----
    metrics = {
        "train_loss": avg_train_loss,
        "train_recon_loss": avg_train_recon_loss,
        "train_kl_loss": avg_train_kl_loss,
        "train_loss_beta_1": avg_train_loss_beta_1,
        "train_kl_loss_beta_1": avg_train_kl_loss_beta_1,
        "val_loss": avg_val_loss,
        "val_recon_loss": avg_val_recon_loss,
        "val_kl_loss": avg_val_kl_loss,
        "learning_rate": current_lr,
        "kl_beta": kl_beta,
    }

    log_to_txt(epoch, metrics)
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        filename = (
            f"{model_name}_batch_{batch_size}_epochs_{num_epochs}"
            f"_zdim_{z_dim}_{kl_beta_annealer.schedule}_{beta_end}_stall_end.pth"
        )

        save_path = os.path.join(BASE_DIR, "model", "weights", filename)
        torch.save(model.state_dict(), save_path)
        logging.info(f"New best model saved at epoch {epoch+1} with val loss {avg_val_loss:.4f}")


logging.info("Training complete")