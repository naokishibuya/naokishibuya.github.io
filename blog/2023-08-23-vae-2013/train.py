import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        
        # Estimate mean and log variance
        self.fc1 = nn.Linear(64*7*7, 400)  # 7x7 feature maps
        self.fc2_mean = nn.Linear(400, latent_dim)
        self.fc2_logvar = nn.Linear(400, latent_dim)
    
    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # Feature extraction
        x = self.feature_extractor(x)

        # Estimate mean and log variance
        x = F.relu(self.fc1(x))
        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        
        # Transform latent variables to a suitable shape for later upsampling
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64*7*7),
            nn.ReLU(),
        )
        
        # Upsampling with transposed convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # Ensuring output is in [0,1]
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Transform latent variables to a suitable shape
        z = self.fc(z)

        # Reshape z to (batch_size, 64, 7, 7)
        z = z.view(z.size(0), 64, 7, 7)

        # Upsampling for reconstruction
        x_recon = self.decoder(z)
        return x_recon


class VAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        
        # Instantiate the Encoder and Decoder
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from the latent space."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple:
        # Pass the input through the encoder
        mu, logvar = self.encoder(x)
        
        # Reparameterization step
        z = self.reparameterize(mu, logvar)
        
        # Pass the latent vector through the decoder
        x_reconstructed = self.decoder(z)
        
        return x_reconstructed, mu, logvar
    

def loss_function(recon_x, x, mu, logvar):
    """Compute the VAE loss."""

    # Reconstruction loss: explicitly summing over all dimensions
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence loss (regularization term)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    batch_size = x.size(0)
    return (recon_loss + kld_loss)/batch_size


def main():
    # Set device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print('Using {} device'.format(device))
    device = torch.device(device)

    # Load data
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True,
        transform=transform,
        download=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize the VAE and optimizer
    model = VAE(latent_dim=2).to(device)
    model.train()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1.0e-3)

    # Train for multiple epochs
    for epoch in range(100):
        train_loss = 0

        # Training loop
        for batch_idx, (data, _) in enumerate(train_loader):
            # We only use images not labels
            data = data.to(device)
            
            # Forward pass
            recon_batch, mu, logvar = model(data)
            
            # Backward pass
            optimizer.zero_grad()
            loss = loss_function(recon_batch, data, mu, logvar)        
            loss.backward()
            optimizer.step()

            # Accumulate the loss for logging
            train_loss += loss.item()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{:5d}/{:5d} ({:2.0f}%)] Loss: {:8.4f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))

        average_loss = train_loss / len(train_loader.dataset)
        print('Epoch: {} Average loss: {:.4f}'.format(epoch, average_loss))

    # Save the model
    model_path = './vae_model.pth'
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main()
