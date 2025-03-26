# Style-Gan-3

### Installation and Environment Setup
#### Clone the StyleGAN3 repository
```
git clone https://github.com/NVlabs/stylegan3.git
cd stylegan3
```

##### Install dependencies
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
pip install click requests tqdm numpy pyspng ninja imageio-ffmpeg pillow
pip install scipy tensorboard
```

#### Optional but recommended
```
pip install einops  # For efficient tensor operations
```
 
### Dataset Preparation
Organize your dataset: Place your Van Gogh images in a directory, e.g., datasets/van_gogh/
###Resize and convert images to StyleGAN format
```
python dataset_tool.py --source=datasets/van_gogh --dest=van_gogh_dataset --resolution=1024x1024
```

### Training the Style Gan-3 Model
#### Train the model with path length regularization
```
python train.py --outdir=training-runs --cfg=stylegan3-t \
    --data=van_gogh_dataset --gpus=1 --batch=32 --gamma=8.2 --mirror=1 \
    --kimg=10000 --snap=10 --metrics=fid50k_full
```

### Style-Mixing
Once training is complete, you can perform style mixing. This applies the Van Gogh style to your input image.

🔹 Style mixing script: style_mix.py

```python
import torch
import dnnlib
import legacy
import numpy as np
from PIL import Image
import imageio


### Load pretrained StyleGAN3 model
network_pkl = "training-runs/network-snapshot-00010.pkl"  # Replace with the latest snapshot
device = torch.device('cuda')

print('Loading networks from "%s"...' % network_pkl)
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)  # Load the generator

def load_image(img_path, resolution=1024):
    """Load and preprocess input image"""
    img = Image.open(img_path).resize((resolution, resolution))
    img = np.array(img).transpose(2, 0, 1) / 255.0
    img = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0)
    return img

def generate_latents(G, seeds):
    """Generate latent vectors from seeds"""
    latents = []
    for seed in seeds:
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        latents.append(z)
    return latents

def style_mix(G, input_img, style_seeds, resolution=1024):
    """Apply style mixing"""
    # Load input image and resize
    img = load_image(input_img, resolution)

    # Generate latent vectors for style mixing
    w1 = generate_latents(G, [style_seeds[0]])[0]
    w2 = generate_latents(G, [style_seeds[1]])[0]

    # Apply style mixing (50% from each style)
    ws = torch.lerp(w1, w2, 0.5).unsqueeze(0)

    # Generate the styled image
    img_out = G.synthesis(ws, noise_mode='const')[0]
    img_out = (img_out.clamp(-1, 1) + 1) * (255 / 2)
    img_out = img_out.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    # Save output
    output_path = "output/van_gogh_style.png"
    imageio.imsave(output_path, img_out)
    print(f"Saved styled image at {output_path}")
```


'''

### Example usage
```
style_mix(G, "path/to/your/input.jpg", style_seeds=[42, 100])
```

### 🛠️ Running the Style Mixing Script
To apply Van Gogh styling:
```
python style_mix.py
```












