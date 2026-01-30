import os
import torch
from PIL import Image
from .hcvae import HierarchicalCVAE
from .imgs_cond_dataset import CVAEDataset
from torchvision import transforms
from .normalizer import Normalizer

class CVAE_Efficient():
    def __init__(self, device: torch.device, model_path: str):
        self.device = device
        self.model_path = model_path

        self.batch_size = 1024
        self.img_dim = 128
        self.frame_size = 3
        self.num_workers = 16  # for data loading
        self.log_every = 10
        
        self.model = self.initialize_model(weights_path=model_path, frame_size=self.frame_size, img_dim=self.img_dim, device=device)
        self.imgs_transforms = self.initialize_transforms(img_dim=self.img_dim)

    def initialize_model(self, weights_path: str, frame_size: int, img_dim: int, device:torch.device)->HierarchicalCVAE:
        model = HierarchicalCVAE(img_channels=frame_size,img_size=img_dim, attn=True)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model

    def initialize_transforms(self, img_dim:int)->transforms.Compose:
        imgs_transforms = transforms.Compose([
                    transforms.Resize((img_dim, img_dim)),
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
        target_mean    = [ 4.69328707, -0.03879964, 10.74773858]
        target_std_dev = [0.67665561, 0.23729723, 3.20131289]
        
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
