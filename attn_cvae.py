import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


def reconstruction_kld(recon, target, kl_values, beta=1.0):
    
    recon_loss = F.mse_loss(recon, target)
    kl_weights = [0.4, 0.3, 0.2, 0.1]

    kl_loss = 0.0
    for kl, w in zip(kl_values, kl_weights):
        kl_loss += w * kl

    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss
    

def kl_gaussian(mu_q, logvar_q, mu_p, logvar_p):
    kl = 0.5 * (
        logvar_p - logvar_q +
        (torch.exp(logvar_q) + (mu_q - mu_p)**2) / torch.exp(logvar_p)
        - 1
    )

    # Sum over latent dimension (dim=1)
    kl = kl.sum(dim=1)
    
    return kl.mean()     


class ResNetTokenizer(nn.Module):
    def __init__(self, img_size=128, hidden_dim=64, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.feat_dim = 512
        self.hidden_dim = hidden_dim
        self.tokens_h = img_size // 32
        self.tokens_w = img_size // 32
        self.num_tokens = self.tokens_h * self.tokens_w  # e.g., 4 x 4 = 16

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, hidden_dim))

        self.proj = nn.Linear(self.feat_dim, hidden_dim)

    def forward(self, x):
        B = x.size(0)

        feats = self.backbone(x)  # (B, 512, H', W')
        B, C, Hf, Wf = feats.shape  # should equal tokens_h × tokens_w

        tokens = feats.permute(0, 2, 3, 1).reshape(B, Hf * Wf, C)

        tokens = self.proj(tokens) + self.pos_embed

        return tokens


class VectorTokenizer(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, seq_len=8):
        super().__init__()
        self.seq_len = seq_len
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * seq_len),  
            nn.ReLU()
        )
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, hidden_dim))
        
    def forward(self, x):
        x = self.mlp(x)                             # (B, hidden_dim * seq_len)
        x = x.view(x.shape[0], self.seq_len, -1)    # (B, seq_len, hidden_dim)
        x = x + self.pos_embed
        return x                                    # (B, seq_len, hidden_dim)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, q, k, v):
        
        q = self.norm_q(q)
        k = self.norm_kv(k)
        v = self.norm_kv(v)
        h, _ = self.attn(q, k, v)
        x = q + h
        h = self.mlp(self.norm2(x))
        x = x + h

        return x
    

class Head(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (B, seq_len, hidden_dim)
        x_pooled = x.mean(dim=1)  # (B, hidden_dim)
        mu = self.mu(x_pooled)          # (B, latent_dim)
        logvar = self.logvar(x_pooled)  # (B, latent_dim)
        logvar = torch.clamp(logvar, -10, 10)
        return mu, logvar


class LatentProjection(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(latent_dim, hidden_dim)

    def forward(self, z):
        return self.proj(z)


class EncoderLevel(nn.Module):
    def __init__(self, hidden_dim_in, hidden_dim_out):
        super().__init__()
        self.attn = TransformerBlock(hidden_dim_in) 
        self.proj = nn.Linear(hidden_dim_in, hidden_dim_out)

    def forward(self, x, cond):
        h = self.attn(q=x, k=cond, v=cond)
        h = self.proj(h)
        return h


class FinalHead(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.01)
        self.attn_pool = nn.MultiheadAttention(embed_dim=hidden_dim, 
                                               num_heads=num_heads, 
                                               batch_first=True)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        B, L, H = x.shape
        query = self.query.expand(B, -1, -1)                       # (B, 1, H)
        attn_out, _ = self.attn_pool(query=query, key=x, value=x)  # (B, 1, H)
        pooled = attn_out.squeeze(1)                               # (B, H)
        out = self.proj(pooled)                                    # (B, output_dim)
        return out


class DecoderLevel(nn.Module):
    def __init__(self, hidden_dim_in, hidden_dim_out):
        super().__init__()
        self.attn = TransformerBlock(hidden_dim_in) 
        self.proj = nn.Linear(hidden_dim_in, hidden_dim_out)
        self.final_head = FinalHead(hidden_dim_out, output_dim=3)

    def forward(self, x, cond):
        h = self.attn(q=x, k=cond, v=cond)
        h = self.proj(h)
        out = self.final_head(h)
        return out


class attnCVAE(nn.Module):
    def __init__(self, hidden_dim=32, input_dim=3, img_channels=3,  img_size=256, latent_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.patch_size = 16
        #self.seq_len    = (img_size // self.patch_size) ** 2 
        self.seq_len    = 16
        
        # Conditioning image tokenizers
        self.cond = ResNetTokenizer(img_size=img_size, hidden_dim=hidden_dim)

        # Input tokenizer
        self.input_tokenizer = VectorTokenizer(input_dim=input_dim, hidden_dim=hidden_dim, seq_len=self.seq_len)

        # Encoder
        self.enc = EncoderLevel(hidden_dim, hidden_dim)      

        # Posterior head
        self.pos = Head(hidden_dim, latent_dim)    

        # Prior head
        self.pri = Head(hidden_dim, latent_dim) 

        # Decoder    
        self.dec =  DecoderLevel(hidden_dim, hidden_dim)  

        # Latent Projection
        self.lat = VectorTokenizer(input_dim=latent_dim, hidden_dim=hidden_dim, seq_len=self.seq_len)  

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    
    def forward(self, x, c):

        c = self.cond(c)
        x = self.input_tokenizer(x)

        h = self.enc(x, c)
        
        mu, logvar = self.pos(h)
        mu_p, logvar_p = self.pri(c)
        z = self.reparameterize(mu, logvar)
        q = self.lat(z)
        out = self.dec(q, c)
        
        kl = kl_gaussian(mu, logvar, mu_p, logvar_p)
        
        return out, [kl, kl, kl, kl]


    @torch.no_grad()
    def generate(self, c, batch=1, device="cpu"):
        c = c.to(device).expand(batch, -1, -1, -1)
        
        c = self.cond(c)
        mu, logvar = self.pri(c)
        z = self.reparameterize(mu, logvar) 
        q = self.lat(z)
        out = self.dec(q, c)

        return out


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_size = 256
    num_workers = 16
    batch_size = 16
    history = 3

    import os
    import torchvision.transforms as transforms
    from cvae.model.imgs_cond_dataset import CVAEDataset
    from tqdm import tqdm

    model = HierarchicalCVAE(hidden_dim=32, input_dim=3, img_channels=history,  img_size=img_size, attn=True).to(device)
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    val_imgs_root = os.path.join(BASE_DIR, 'data/data_v2/val/imgs/')
    targets_val_path = os.path.join(BASE_DIR, 'data/data_v2/val/x_validation.parquet')

    imgs_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(lambda x: 1.0 - x),
        transforms.Normalize( mean=[0.5], std=[0.5])
    ])
    
    val_dataset = CVAEDataset(
        targets_path=targets_val_path,
        image_root=val_imgs_root,
        mode='val',
        image_transform=imgs_transforms,
        normalize=True,
        num_workers=num_workers,
        history=history
    )

    #
    x, img = val_dataset[0]  
    x = x.unsqueeze(0).to(device)   
    img = img.unsqueeze(0).to(device)  

    model.eval()
    with torch.no_grad():
        output, kl = model(x, img)
        output = model.generate(img, device=device)