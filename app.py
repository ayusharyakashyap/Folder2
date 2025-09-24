# app.py
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cpu")

class CVAE(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(CVAE, self).__init__()
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

latent_dim = 100
model = CVAE(latent_dim).to(device)
model.load_state_dict(torch.load("cvae.pth", map_location=device))
model.eval()

def one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels]

st.title("Handwritten Digit Generator (0-9)")

digit = st.selectbox("Select a digit to generate:", list(range(10)))

if st.button("Generate"):
    with torch.no_grad():
        z = torch.randn(5, latent_dim)
        labels = torch.full((5,), digit, dtype=torch.long)
        labels_oh = one_hot(labels)
        gen_imgs = model.decode(z, labels_oh).view(-1, 28, 28)
        gen_imgs = gen_imgs.numpy()

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for img, ax in zip(gen_imgs, axes):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    st.pyplot(fig)
