# Efficient sampling through a Hierarchical Conditional Variational Autoencoder (HCVAE)

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

Remember to de-normalize the output of the model, through this function:
```bash
parameters = test_dataset.normalizer.inverse_transform_targets(parameters_normalized.cpu().numpy())
``` 
If you run ```inference.py```, and if you have the test dataset saved in this path: 
```bash
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

test_imgs_root = os.path.join(BASE_DIR, 'data/data_v2/test/imgs/')
targets_test_path = os.path.join(BASE_DIR, 'data/data_v2/test/sampled_vars.parquet')
```  

you should get the output:
```bash
[INFO] Using device: cuda
[INFO] [TEST] Loading normalizer...
[INFO] Normalizer loaded from cvae/model/weights/
[INFO] Target mean: [ 4.69328707 -0.03879964 10.74773858]
[INFO] Target std: [0.67665561 0.23729723 3.20131289]
[INFO] [TEST] Data normalized
[INFO] Preloading 12911 unique images using 16 workers...
Loading images: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12911/12911 [00:15<00:00, 821.89it/s]
[INFO] Preloading complete. Cache contains 12911 images.
batch reconstruction + kl loss: 0.0718
batch reconstruction + kl loss: 0.0691
batch reconstruction + kl loss: 0.0714
batch reconstruction + kl loss: 0.0707
batch reconstruction + kl loss: 0.0708
batch reconstruction + kl loss: 0.0710
batch reconstruction + kl loss: 0.0699
batch reconstruction + kl loss: 0.0700
batch reconstruction + kl loss: 0.0701
batch reconstruction + kl loss: 0.0716
batch reconstruction + kl loss: 0.0729
batch reconstruction + kl loss: 0.0715
batch reconstruction + kl loss: 0.0706
Total reconstruction + kl loss: 0.0709
Generated sample: [[ 4.68577862e+00 -6.67941943e-02  1.09836121e+01]
 [ 4.88425589e+00 -5.46252429e-02  1.20996017e+01]
 [ 4.95344210e+00  1.13169244e-02  1.22437105e+01]
 [ 4.94653463e+00  3.82092036e-02  1.14626055e+01]
 [ 4.88106728e+00  4.10081409e-02  1.30424747e+01]
 [ 4.76561022e+00  1.80595294e-01  9.56018162e+00]
 [ 4.93080902e+00  2.26294007e-02  1.16201830e+01]
 [ 4.93429184e+00  3.28624435e-02  1.19620123e+01]
 [ 3.99375820e+00 -4.80793938e-02  9.46704292e+00]
 [ 4.17344618e+00  1.40651306e-02  1.11908112e+01]]
``` 
