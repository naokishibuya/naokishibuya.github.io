import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# vae.py から VAE をインポート
from train import VAE

# VAE インスタンスを作成し、保存した重みを読み込む
latent_dim = 2

model = VAE(latent_dim)
model.load_state_dict(torch.load('./vae_model.pth'))
model.eval()


# 2次元の潜在空間のグリッドを作成し、画像を生成
steps = 14
latent_values = torch.linspace(-1.5, 1.5, steps)
grid_z = torch.tensor([[z1, z2] for z1 in latent_values for z2 in latent_values])

with torch.no_grad():
    images = model.decoder(grid_z)


# 生成した画像を7x7のグリッドに可視化
grid = make_grid(images, nrow=steps, padding=1, pad_value=1)
grid = grid.permute(1, 2, 0)

plt.imshow(grid)
plt.axis('off')
plt.title('2D Latent Space Exploration')
plt.tight_layout()
plt.show()
