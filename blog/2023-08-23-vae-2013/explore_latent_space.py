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


# 2. Generate a 2D grid of values in the latent space and generate images
steps = 14
latent_values = torch.linspace(-1.5, 1.5, steps)
grid_z = torch.tensor([[z1, z2] for z1 in latent_values for z2 in latent_values])

with torch.no_grad():
    images = model.decoder(grid_z)


# 3. Visualize the generated images in a 7x7 grid
grid = make_grid(images, nrow=steps, padding=1, pad_value=1)
grid = grid.permute(1, 2, 0)

plt.imshow(grid)
plt.axis('off')
plt.title('2D Latent Space Exploration')
plt.show()
