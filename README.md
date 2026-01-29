#Efficient sampling through a Hierarchical Conditional Variational Autoencoder (HCVAE)

The weights are in the weight folder. 
You can check how to generate samples with the trained model in the ```inference.py``` script:
```bash
# Generate samples:
num_trajectories_to_generate = 10
with torch.inference_mode():
    imgs_cond = test_dataset[0][1].unsqueeze(0).to(device) 
    parameters_normalized = model.generate(c=imgs_cond, batch=num_trajectories_to_generate, device=device)
    parameters = test_dataset.normalizer.inverse_transform_targets(parameters_normalized.cpu().numpy())
``` 
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
