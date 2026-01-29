Efficient sampling through a Hierarchical Conditional Variational Autoencoder (HCVAE)
The weights are in the weight folder. 
You can check how the model works in the ```inference.py``` script:
```bash
# Generate samples:
num_trajectories_to_generate = 10
with torch.inference_mode():
    imgs_cond = test_dataset[0][1].unsqueeze(0).to(device) 
    parameters_normalized = model.generate(c=imgs_cond, batch=num_trajectories_to_generate, device=device)
    parameters = test_dataset.normalizer.inverse_transform_targets(parameters_normalized.cpu().numpy())
``` 
