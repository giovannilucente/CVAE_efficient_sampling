import torch
import torch.nn as nn
import torch.nn.functional as F


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

    # Sum over latent dimension (dim=2), mean over sequence dimension (dim=1)
    kl = kl.sum(dim=2).mean(dim=1)
    
    return kl.mean()     


class ImageTokenizer(nn.Module):
    def __init__(self, img_channels=3, img_size=128, patch_size=16, hidden_dim=64):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        
        self.img_channels = img_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        self.num_patches = (img_size // patch_size) ** 2
        
        self.patch_proj = nn.Linear(img_channels * patch_size * patch_size, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim))
    
    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        
        x_unf = x.unfold(2, P, P).unfold(3, P, P)                               #(B, C, H//P, W//P, P, P)
        x_unf = x_unf.contiguous().permute(0, 2, 3, 1, 4, 5).contiguous()       #(B, H//P, W//P, C, P, P)
        B, H_p, W_p, C, P, _ = x_unf.shape
        x_unf = x_unf.view(B, H_p * W_p, C * P * P)                             #(B, n_patches, C*P*P)
        
        tokens = self.patch_proj(x_unf)                                         #(B, n_patches, hidden_dim)
        
        tokens = tokens + self.pos_embed  
        B, N, D = tokens.shape
        tokens = tokens.view(B, N//2, 2, D).mean(dim=2)
        return tokens


class VectorTokenizer(nn.Module):
    def __init__(self, hidden_dim=64, seq_len=8):
        super().__init__()
        self.seq_len = seq_len
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim * seq_len),  
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.mlp(x)                  # (B, hidden_dim * seq_len)
        x = x.view(x.shape[0], self.seq_len, -1)  # (B, seq_len, hidden_dim)
        return x  # (B, seq_len, hidden_dim)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):

        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h

        h = self.mlp(self.norm2(x))
        x = x + h

        return x


class TokenMLPBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        hidden_dim = int(dim * mlp_ratio)

        self.norm = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm(x)
        h = self.mlp(h)
        return x + h 


class PosteriorHead(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        mu = self.mu(x)          # (B, seq_len, latent_dim)
        logvar = self.logvar(x)  # (B, seq_len, latent_dim)
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar



class PriorHead(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        mu = self.mu(x)          # (B, seq_len, latent_dim)
        logvar = self.logvar(x)  # (B, seq_len, latent_dim)
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar



class LatentProjection(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(latent_dim, hidden_dim)

    def forward(self, z):
        return self.proj(z)


class EncoderLevel(nn.Module):
    def __init__(self, hidden_dim_in, hidden_dim_out, attn=False):
        super().__init__()
        self.block = TransformerBlock(hidden_dim_in) if attn==True else TokenMLPBlock(hidden_dim_in)
        self.proj = nn.Linear(hidden_dim_in, hidden_dim_out)

    def forward(self, x):
        h = self.block(x)
        B, N, D = h.shape
        h = h.view(B, N//2, 2, D).mean(dim=2)
        h = self.proj(h)
        return h


class DecoderLevel(nn.Module):
    def __init__(self, hidden_dim_in, hidden_dim_out, attn=False):
        super().__init__()
        self.block = TransformerBlock(hidden_dim_in) if attn==True else TokenMLPBlock(hidden_dim_in)
        self.proj = nn.Linear(hidden_dim_in, hidden_dim_out)

    def forward(self, x):
        h = self.block(x)
        B, N, D = h.shape
        h = h.view(B, N//2, 2, D).mean(dim=2)
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


class HierarchicalCVAE(nn.Module):
    def __init__(self, hidden_dim=32, input_dim=3, img_channels=3,  img_size=256, latent_dim=64, attn=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.patch_size = 16
        self.seq_len           = (img_size // self.patch_size) ** 2 // 2
        self.top_latent_length = (img_size // self.patch_size) ** 2 // 2

        self.top_latent = nn.Parameter(0.01 * torch.randn(1, self.top_latent_length, hidden_dim*8))

        # Conditioning image tokenizers
        self.cond1 = ImageTokenizer(img_channels=img_channels, img_size=img_size, hidden_dim=hidden_dim,   patch_size=self.patch_size)
        self.cond2 = ImageTokenizer(img_channels=img_channels, img_size=img_size, hidden_dim=hidden_dim*2,  patch_size=self.patch_size)
        self.cond3 = ImageTokenizer(img_channels=img_channels, img_size=img_size, hidden_dim=hidden_dim*4, patch_size=self.patch_size)
        self.cond4 = ImageTokenizer(img_channels=img_channels, img_size=img_size, hidden_dim=hidden_dim*8, patch_size=self.patch_size)

        # Input tokenizer
        self.input_tokenizer = VectorTokenizer(hidden_dim=hidden_dim, seq_len=self.seq_len)

        # Encoder
        self.enc1 = EncoderLevel(hidden_dim,    hidden_dim,   attn=attn)       
        self.enc2 = EncoderLevel(hidden_dim,    hidden_dim*2, attn=attn)     
        self.enc3 = EncoderLevel(hidden_dim*2,  hidden_dim*4, attn=attn)     
        self.enc4 = EncoderLevel(hidden_dim*4,  hidden_dim*8, attn=attn)     

        # Posterior head
        self.pos1 = PosteriorHead(hidden_dim,   latent_dim)      
        self.pos2 = PosteriorHead(hidden_dim*2, latent_dim)    
        self.pos3 = PosteriorHead(hidden_dim*4, latent_dim)    
        self.pos4 = PosteriorHead(hidden_dim*8, latent_dim)    

        # Prior head
        self.pri1 = PriorHead(hidden_dim,   latent_dim)      
        self.pri2 = PriorHead(hidden_dim*2, latent_dim)    
        self.pri3 = PriorHead(hidden_dim*4, latent_dim)

        # Decoder 
        self.dec3 =  DecoderLevel(hidden_dim*8,  hidden_dim*4, attn=attn)    
        self.dec2 =  DecoderLevel(hidden_dim*4,  hidden_dim*2, attn=attn)    
        self.dec1 =  DecoderLevel(hidden_dim*2,  hidden_dim,   attn=attn)    
        
        self.final = FinalHead(hidden_dim, input_dim)  

        # Latent Projection
        self.lat1 = LatentProjection(latent_dim, hidden_dim)   
        self.lat2 = LatentProjection(latent_dim, hidden_dim*2)
        self.lat3 = LatentProjection(latent_dim, hidden_dim*4)
        self.lat4 = LatentProjection(latent_dim, hidden_dim*8)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    
    def forward(self, x, c):

        batch = x.shape[0]
        d4 = self.top_latent.expand(batch, -1, -1)

        c1 = self.cond1(c)
        c2 = self.cond2(c)
        c3 = self.cond3(c)
        c4 = self.cond4(c)

        x1 = self.input_tokenizer(x)

        h1 = self.enc1(torch.cat([x1, c1], dim=1))
        h2 = self.enc2(torch.cat([h1, c1], dim=1))
        h3 = self.enc3(torch.cat([h2, c2], dim=1))
        h4 = self.enc4(torch.cat([h3, c3], dim=1))
        
        mu4, logvar4 = self.pos4(h4)
        mu4_p     = torch.zeros_like(mu4)
        logvar4_p = torch.zeros_like(logvar4)
        z4 = self.reparameterize(mu4, logvar4)
        q4 = self.lat4(z4)
        w4 = d4 + 0.1 * q4
        d3 = self.dec3(torch.cat([w4, c4], dim=1))

        k3 = h3 + 0.1 * d3
        mu3, logvar3 = self.pos3(k3)
        mu3_p, logvar3_p = self.pri3(d3)
        z3 = self.reparameterize(mu3, logvar3)
        q3 = self.lat3(z3)
        w3 = d3 + 0.1 * q3
        d2 = self.dec2(torch.cat([w3, c3], dim=1))

        k2 = h2 + 0.1 * d2
        mu2, logvar2 = self.pos2(k2)
        mu2_p, logvar2_p = self.pri2(d2)
        z2 = self.reparameterize(mu2, logvar2)
        q2 = self.lat2(z2)
        w2 = d2 + 0.1 * q2
        d1 = self.dec1(torch.cat([w2, c2], dim=1))

        k1 = h1 + 0.1 * d1
        mu1, logvar1 = self.pos1(k1)
        mu1_p, logvar1_p = self.pri1(d1)
        z1 = self.reparameterize(mu1, logvar1)
        q1 = self.lat1(z1)
        w1 = d1 + 0.1 * q1
        out = self.final(torch.cat([w1, c1], dim=1))
        
        
        kl1 = kl_gaussian(mu1, logvar1, mu1_p, logvar1_p)
        kl2 = kl_gaussian(mu2, logvar2, mu2_p, logvar2_p)
        kl3 = kl_gaussian(mu3, logvar3, mu3_p, logvar3_p)
        kl4 = kl_gaussian(mu4, logvar4, mu4_p, logvar4_p)
        
        return out, [kl1, kl2, kl3, kl4]


    @torch.no_grad()
    def generate(self, c, batch=1, device="cpu"):
        c = c.to(device).expand(batch, -1, -1, -1)
        d4 = self.top_latent.expand(batch, -1, -1)

        c1 = self.cond1(c)
        c2 = self.cond2(c)
        c3 = self.cond3(c)
        c4 = self.cond4(c)

        z4 = torch.randn(batch, self.seq_len, self.latent_dim).to(device)
        q4 = self.lat4(z4)
        w4 = d4 + q4
        d3 = self.dec3(torch.cat([w4, c4], dim=1))

        mu3, logvar3 = self.pri3(d3)
        z3 = self.reparameterize(mu3, logvar3)
        q3 = self.lat3(z3)
        w3 = d3 + q3
        d2 = self.dec2(torch.cat([w3, c3], dim=1))

        mu2, logvar2 = self.pri2(d2)
        z2 = self.reparameterize(mu2, logvar2)
        q2 = self.lat2(z2)
        w2 = d2 + q2
        d1 = self.dec1(torch.cat([w2, c2], dim=1))

        mu1, logvar1 = self.pri1(d1)
        z1 = self.reparameterize(mu1, logvar1)
        q1 = self.lat1(z1)
        w1 = d1 + q1
        out = self.final(torch.cat([w1, c1], dim=1))

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
        transforms.Resize((img_dim, img_dim)),
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
            
