import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Assuming VAE is defined in the train.py
from train import VAE


# 1. Create a new VAE instance and load the saved weights
latent_dim = 2

model = VAE(latent_dim)
model.load_state_dict(torch.load('./vae_model.pth'))
model.eval()


# 2. Sample from the latent space (the standard normal) and generate images
num_samples = 49
torch.manual_seed(42)
z = torch.randn(num_samples, latent_dim)

with torch.no_grad():
    images = model.decoder(z)


# 3. Visualize the generated images in a grid
grid = make_grid(images, nrow=7, padding=1, pad_value=1)
grid = grid.permute(1, 2, 0)

plt.imshow(grid)
plt.axis('off')
plt.title('Randomly Generated Images')
plt.show()