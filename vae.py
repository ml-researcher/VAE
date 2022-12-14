import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torchvision.utils import save_image

batch_size = 100
sample_freq = 5
epoch = 50
dim_z = 10

train_dataset = datasets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist', train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class Enc(nn.Module):
    def __init__(self, dim_in, dim1, dim2, dim_z):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim1)
        self.fc2 = nn.Linear(dim1, dim2)
        self.fcz1 = nn.Linear(dim2, dim_z)
        self.fcz2 = nn.Linear(dim2, dim_z)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu, log_var = self.fcz1(x), self.fcz2(x)
        return mu, log_var
    def reparam(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        t = torch.randn_like(sigma)
        return mu + t * sigma

class Dec(nn.Module):
    def __init__(self, dim_in, dim1, dim2, dim_z):
        super().__init__()
        self.fc1 = nn.Linear(dim_z, dim2)
        self.fc2 = nn.Linear(dim2, dim1)
        self.fc3 = nn.Linear(dim1, dim_in)
    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x.view(-1, 1, 28, 28)

class VAE(nn.Module):
    def __init__(self, dim_in, dim1, dim2, dim_z):
        super().__init__()
        self.encoder = Enc(dim_in, dim1, dim2, dim_z)
        self.decoder = Dec(dim_in, dim1, dim2, dim_z)
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.encoder.reparam(mu, log_var)
        xp = self.decoder(z)
        return xp, mu, log_var

def loss_func(x, xps, mu, log_var):
    """
    bce?????????????????????z????????????
    """
    bces = [F.binary_cross_entropy(xp, x, reduction='sum') for xp in xps]
    bce = sum(bces) / len(bces)
    kl = 0.5 * torch.sum(-log_var + mu**2 + torch.exp(log_var))
    return bce + kl

def train_one_epoch(e, model, optimizer):
    model.train()
    pbar = tqdm(train_loader)
    for data, _ in pbar:
        if torch.cuda.is_available():
            data = data.cuda()
        optimizer.zero_grad()
        mu, log_var = model.encoder(data)
        zs = [model.encoder.reparam(mu, log_var) for _ in range(sample_freq)]
        xps = [model.decoder(z) for z in zs]
        loss = loss_func(data, xps, mu, log_var) / data.size(0)
        loss.backward()
        optimizer.step()
        pbar.set_description("epoch {} train loss: {:.2f}".format(e, loss.item()))

def test_one_epoch(e, model):
    model.eval()
    pbar = tqdm(test_loader)
    total_loss = 0.
    with torch.no_grad():
        for data, _ in pbar:
            if torch.cuda.is_available():
                data = data.cuda()
            mu, log_var = model.encoder(data)
            zs = [model.encoder.reparam(mu, log_var) for _ in range(sample_freq)]
            xps = [model.decoder(z) for z in zs]
            loss = loss_func(data, xps, mu, log_var).item()
            total_loss += loss
            pbar.set_description("epoch {} test loss: {:.2f}".format(e, loss / data.size(0)))
    print('test average loss: {:.2f}'.format(total_loss / len(test_loader.dataset)))

def sampling(e, model):
    model.eval()
    z = torch.randn(64, dim_z)
    if torch.cuda.is_available():
        z = z.cuda()
    with torch.no_grad():
        sample = model.decoder(z)
        # ????????????????????????pdf????????????
        sample[sample>0.5] = 1.
        sample[sample<0.5] = 0.
        save_image(sample, './samples/sample_{}'.format(e) + '.png')

if __name__ == '__main__':
    model = VAE(28*28, 256, 128, dim_z)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters())
    os.makedirs('./samples', exist_ok=True)
    for e in range(epoch):
        train_one_epoch(e, model, optimizer)
        test_one_epoch(e, model)
        sampling(e, model)