# # train_cvae.py
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # CVAE Model
# class CVAE(nn.Module):
#     def __init__(self, latent_dim=50, num_classes=10):
#         super(CVAE, self).__init__()
#         self.num_classes = num_classes
#         self.fc1 = nn.Linear(28*28 + num_classes, 400)
#         self.fc21 = nn.Linear(400, latent_dim)
#         self.fc22 = nn.Linear(400, latent_dim)
#         self.fc3 = nn.Linear(latent_dim + num_classes, 400)
#         self.fc3_bn = nn.BatchNorm1d(400)  # ✅ BatchNorm correctly placed inside __init__
#         self.fc4 = nn.Linear(400, 28*28)

#     def encode(self, x, labels):
#         x = torch.cat([x, labels], 1)
#         h1 = torch.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)  # mu, logvar

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std

#     def decode(self, z, labels):
#         z = torch.cat([z, labels], 1)
#         h3 = torch.relu(self.fc3_bn(self.fc3(z)))  # ✅ BatchNorm used here
#         return torch.sigmoid(self.fc4(h3))

#     def forward(self, x, labels):
#         mu, logvar = self.encode(x, labels)
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z, labels), mu, logvar

# # Loss Function
# def loss_function(recon_x, x, mu, logvar):
#     BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return BCE + KLD

# # Data
# transform = transforms.Compose([transforms.ToTensor()])
# train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# # Model, Optimizer
# latent_dim = 50
# model = CVAE(latent_dim).to(device)
# optimizer = optim.AdamW(model.parameters(), lr=1e-3)  # ✅ AdamW used

# # One-hot labels
# def one_hot(labels, num_classes=10):
#     return torch.eye(num_classes)[labels].to(device)

# # Training Loop
# epochs = 20  # ✅ Increase to 20
# for epoch in range(epochs):
#     model.train()
#     train_loss = 0
#     for batch_idx, (data, labels) in enumerate(train_loader):
#         data = data.view(-1, 28*28).to(device)
#         labels_oh = one_hot(labels)
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = model(data, labels_oh)
#         loss = loss_function(recon_batch, data, mu, logvar)
#         loss.backward()
#         train_loss += loss.item()
#         optimizer.step()
#     print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader.dataset):.4f}')

# torch.save(model.state_dict(), "cvae.pth")
# print("✅ Training complete. Model saved as cvae.pth.")



# train_cvae.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CVAE(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(CVAE, self).__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(28*28 + num_classes, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc21 = nn.Linear(256, latent_dim)
        self.fc22 = nn.Linear(256, latent_dim)

        self.fc3 = nn.Linear(latent_dim + num_classes, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 28*28)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def encode(self, x, labels):
        x = torch.cat([x, labels], 1)
        h1 = self.leakyrelu(self.fc1(x))
        h2 = self.leakyrelu(self.fc2(h1))
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, labels):
        z = torch.cat([z, labels], 1)
        h3 = self.leakyrelu(self.fc3_bn(self.fc3(z)))
        h4 = self.leakyrelu(self.fc4(h3))
        return torch.sigmoid(self.fc5(h4))

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

latent_dim = 100
model = CVAE(latent_dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

def one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels].to(device)

epochs = 20
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.view(-1, 28*28).to(device)
        labels_oh = one_hot(labels)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, labels_oh)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader.dataset):.4f}')

torch.save(model.state_dict(), "cvae.pth")
print("✅ Training complete. Model saved as cvae.pth.")
