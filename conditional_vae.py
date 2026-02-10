import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights

class CNNFeatureExtractor(nn.Module):
    def __init__(self, output_dim=64, in_channels=3):
        super().__init__()

        from torchvision.models import resnet18
        res = resnet18(weights=None)

        # Modify first conv for multi-channel input
        res.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=res.conv1.out_channels,
            kernel_size=res.conv1.kernel_size,
            stride=res.conv1.stride,
            padding=res.conv1.padding,
            bias=False
        )

        # Remove final classifier
        self.features = nn.Sequential(*list(res.children())[:-1])
        resnet_output_dim = 512

        # Projection head 
        self.projection = nn.Sequential(
            nn.Linear(resnet_output_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.projection(x)
        return x

def kl_gaussian(mu, logvar):
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    return kl_loss.mean()
    
    
class CVAE(nn.Module):
    def __init__(self, X_dim=3, z_dim=64, h_Q_dim=512, h_P_dim=512, cnn_feature_dim=64, history=3):
        super(CVAE, self).__init__()
        self.z_dim = z_dim
        
        # CNN for image feature extraction with ResNet18 backbone
        self.cnn_extractor = CNNFeatureExtractor(output_dim=cnn_feature_dim, in_channels=history)
        
        # Encoder (Q network) with BatchNorm and Dropout
        self.encoder = nn.Sequential(
            nn.Linear(X_dim + cnn_feature_dim, h_Q_dim),
            nn.BatchNorm1d(h_Q_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h_Q_dim, h_Q_dim // 2),
            nn.BatchNorm1d(h_Q_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h_Q_dim // 2, h_Q_dim // 4),
            nn.BatchNorm1d(h_Q_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.encoder_fc_mu = nn.Linear(h_Q_dim // 4, z_dim)
        self.encoder_fc_logvar = nn.Linear(h_Q_dim // 4, z_dim)
        
        # Decoder (P network) - Symmetric to encoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + cnn_feature_dim, h_P_dim // 4),
            nn.BatchNorm1d(h_P_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h_P_dim // 4, h_P_dim // 2),
            nn.BatchNorm1d(h_P_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h_P_dim // 2, h_P_dim),
            nn.BatchNorm1d(h_P_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h_P_dim, X_dim)  # No activation on output
        )

    def encode(self, x, imgs):
        imgs_features = self.cnn_extractor(imgs)
        xc = torch.cat([x, imgs_features], dim=1)
        
        h = self.encoder(xc)
        mu = self.encoder_fc_mu(h)
        logvar = self.encoder_fc_logvar(h)

        return mu, logvar, imgs_features

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, img_features):
        zc = torch.cat([z, img_features], dim=1)
        y = self.decoder(zc)

        return y

    def forward(self, x, c):
        mu, logvar, imgs_features = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        y = self.decode(z, imgs_features)
        kl = kl_gaussian(mu, logvar)
        return y, [kl, kl, kl, kl]

    @torch.no_grad()
    def generate(self, c, batch=1, device='cpu'):

        imgs_features = self.cnn_extractor(c)                             # shape: (1, cnn_feature_dim)
        z = torch.randn(batch, self.z_dim, device=device)

        imgs_expanded = imgs_features.repeat(batch, 1)                # (batch, cnn_feature_dim)
        zc = torch.cat([z, imgs_expanded], dim=1)

        y = self.decoder(zc)                                                # (num_samples, X_dim)

        return y



