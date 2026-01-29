# Efficient sampling through a Hierarchical Conditional Variational Autoencoder (HCVAE)

The weights are in the weight folder. 
You can check how to generate samples with the trained model in the ```inference.py``` script, by using the function ``` generate_samples(model, imgs_list, num_samples, transformation, normalizer, device)``` :
```bash

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
    target_mean    = [ 4.69328707, -0.03879964, 10.74773858]
    target_std_dev = [0.67665561, 0.23729723, 3.20131289]
    
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

imags = [Image.open(p).convert("RGB") for p in img_paths]
normalizer = Normalizer()
parameters = generate_samples(model, imags, num_samples=5, transformation=imgs_transforms, normalizer=normalizer, device=device)
print(f"Generated samples: {parameters}")
```
Remember to load the normalizer and the transformations needed for the images.
For this model I used a history of 3 BEV images with dimension ``` img_dim=128``` , in black and white, saved in a 3 channel tensor. The images of the dataset need to have these transformations before getting passed to the model:
```bash
imgs_transforms = transforms.Compose([
            transforms.Resize((img_dim, img_dim)),
            transforms.ToTensor(),  
            transforms.Grayscale(num_output_channels=1),
            transforms.Lambda(lambda x: 1.0 - x),
            transforms.Normalize( mean=[0.5], std=[0.5])
        ])
```
So the images are resized with the dimension 128x128 pixels, then transformed in tensors, then transformed in a grayscale, then through the function Lambda inverted (black background and traffic participants white). Finally the images are normalized.

