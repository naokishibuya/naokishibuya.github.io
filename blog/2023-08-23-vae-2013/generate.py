import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# vae.py から VAE をインポート
from vae import VAE


# VAE インスタンスを作成し、保存した重みを読み込む
latent_dim = 2

model = VAE(latent_dim)
model.load_state_dict(torch.load('./vae_model.pth'))
model.eval()


# 潜在空間（標準正規分布）からサンプリングし、画像を生成
num_samples = 49
torch.manual_seed(42)
z = torch.randn(num_samples, latent_dim)

with torch.no_grad():
    images = model.decoder(z)


# 生成した画像をグリッドに可視化
grid = make_grid(images, nrow=7, padding=1, pad_value=1)
grid = grid.permute(1, 2, 0)

plt.imshow(grid)
plt.axis('off')
plt.title('Randomly Generated Images')
plt.show()