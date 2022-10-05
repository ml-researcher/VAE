from functools import reduce
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torchvision.utils import save_image
from layers import VQEmbedding

batch_size = 100
sample_freq = 5
epoch = 50
dim_z = 10
vocab_size = 64

train_dataset = datasets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist', train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class Enc(nn.Module):
    def __init__(self, dim_in, dim1, dim2, dim_z):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim1)
        self.fc2 = nn.Linear(dim1, dim2)
        self.fcz = nn.Linear(dim2, dim_z)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z = self.fcz(x)
        return z.view(-1, dim_z, 1, 1)

class Dec(nn.Module):
    def __init__(self, dim_in, dim1, dim2, dim_z):
        super().__init__()
        self.fc1 = nn.Linear(dim_z, dim2)
        self.fc2 = nn.Linear(dim2, dim1)
        self.fc3 = nn.Linear(dim1, dim_in)
    def forward(self, z):
        z = z.view(-1, dim_z)
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x.view(-1, 1, 28, 28)

class VQVAE(nn.Module):
    def __init__(self, dim_in, dim1, dim2, dim_z):
        super().__init__()
        self.encoder = Enc(dim_in, dim1, dim2, dim_z)
        self.decoder = Dec(dim_in, dim1, dim2, dim_z)
        self.codebook = VQEmbedding(vocab_size, dim_z)
    def forward(self, x):
        z_e = self.encoder(x)
        z_d_no_grad, z_d_with_grad = self.codebook.straight_through(z_e)
        xp = self.decoder(z_d_no_grad)
        return xp, z_e, z_d_with_grad

def loss_func(x, xp, z_e, z_d):
    bce = F.binary_cross_entropy(xp, x, reduce='mean')
    loss_vq = F.mse_loss(z_d, z_e.detach())
    loss_commit = F.mse_loss(z_e, z_d.detach())
    return bce + loss_vq + 0.25 * loss_commit

def train_one_epoch(e, model, optimizer):
    model.train()
    pbar = tqdm(train_loader)
    for data, _ in pbar:
        if torch.cuda.is_available():
            data = data.cuda()
        optimizer.zero_grad()
        xp, z_e, z_d = model(data)
        loss = loss_func(data, xp, z_e, z_d)
        loss.backward()
        optimizer.step()
        pbar.set_description("epoch {} train loss: {:.9f}".format(e, loss.item()))

def test_one_epoch(e, model):
    model.eval()
    pbar = tqdm(test_loader)
    total_loss = 0.
    with torch.no_grad():
        for data, _ in pbar:
            if torch.cuda.is_available():
                data = data.cuda()
            xp, z_e, z_d = model(data)
            loss = loss_func(data, xp, z_e, z_d).item()
            pbar.set_description("epoch {} test loss: {:.9f}".format(e, loss))

def sampling(e, model):
    model.eval()
    z = model.codebook.embedding.weight
    if torch.cuda.is_available():
        z = z.cuda()
    with torch.no_grad():
        sample = model.decoder(z)
        # 贪心采样，在教程pdf里有提到
        sample[sample>0.5] = 1.
        sample[sample<0.5] = 0.
        save_image(sample, './samples/sample_{}'.format(e) + '.png')

if __name__ == '__main__':
    model = VQVAE(28*28, 256, 128, dim_z)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters())
    os.makedirs('./samples', exist_ok=True)
    for e in range(epoch):
        train_one_epoch(e, model, optimizer)
        test_one_epoch(e, model)
        sampling(e, model)