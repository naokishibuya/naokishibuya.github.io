import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        
        # 特徴抽出器
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        
        # 平均と分散を推定
        self.fc1 = nn.Linear(64*7*7, 400)  # 7x7 の特徴マップ
        self.fc2_mean = nn.Linear(400, latent_dim)
        self.fc2_logvar = nn.Linear(400, latent_dim)
    
    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # 特徴量を抽出
        x = self.feature_extractor(x)

        # 平均と分散を推定
        x = F.relu(self.fc1(x))
        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        
        # 潜在変数から特徴マップのサイズに変換
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64*7*7),
            nn.ReLU(),
        )
        
        # 転置畳み込み層で画像のサイズを拡大
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # ピクセル値を [0, 1] にする
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # 潜在変数を特徴マップのサイズに変換
        z = self.fc(z)

        # 特徴マップのシェイプに変換　64x7x7
        z = z.view(z.size(0), 64, 7, 7)

        # 転置畳み込み層で画像のサイズを拡大
        x_recon = self.decoder(z)
        return x_recon


class VAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        
        # エンコーダとデコーダを定義
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # パラメータ変換トリックで潜在変数の値をサンプリング
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple:
        # エンコーダで潜在変数の分布を推定
        mu, logvar = self.encoder(x)
        
        # パラメータ嫌韓トリックで潜在変数の値をサンプリング
        z = self.reparameterize(mu, logvar)
        
        # デコーダで画像を再構築
        x_reconstructed = self.decoder(z)
        
        return x_reconstructed, mu, logvar
    

def loss_function(recon_x, x, mu, logvar):
    # 再生損失
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KLダイバージェンス
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # バッチサイズで割って平均を計算
    batch_size = x.size(0)
    return (recon_loss + kld_loss)/batch_size


def main():
    # デバイスの設定
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print('Using {} device'.format(device))
    device = torch.device(device)

    # MNISTデータセットを読み込む
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True,
        transform=transform,
        download=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # VAEモデルを定義
    model = VAE(latent_dim=2).to(device)
    model.train()

    # オプティマイザを定義
    optimizer = optim.AdamW(model.parameters(), lr=1.0e-3)

    # 訓練ループ
    for epoch in range(100):
        train_loss = 0

        # バッチごとに訓練
        for batch_idx, (data, _) in enumerate(train_loader):
            # デバイスにデータを転送（ラベルは使わない）
            data = data.to(device)
            
            # フィードフォワード
            recon_batch, mu, logvar = model(data)
            
            # 損失を計算して誤差逆伝播
            optimizer.zero_grad()
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()

            # 損失を記録
            train_loss += loss.item()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{:5d}/{:5d} ({:2.0f}%)] Loss: {:8.4f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))

        average_loss = train_loss / len(train_loader.dataset)
        print('Epoch: {} Average loss: {:.4f}'.format(epoch, average_loss))

    # モデルを保存
    model_path = './vae_model.pth'
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main()
