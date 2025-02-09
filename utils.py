import torch
from torch import nn
from torch.autograd import Variable
import pandas as pd
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import transforms

def get_data(batch_size):
    low_resolution_data = ImageFolder('CUB_200_2011/images/', transform=transforms.Compose([
        transforms.Resize((64,64)), transforms.ToTensor()
    ]))
    high_resolution_data = ImageFolder('CUB_200_2011/images/', transform=transforms.Compose([
        transforms.Resize((256,256)), transforms.ToTensor()
    ]))
    e = pd.read_pickle('birds/train/char-CNN-RNN-embeddings.pickle')
    includes = []
    f = open('includes.txt', 'r')
    for x in f.readlines():
        includes.append(int(x))
    low_resolution_dataloader = DataLoader(Subset(low_resolution_data, includes), batch_size=batch_size, shuffle=False)
    high_resolution_dataloader = DataLoader(Subset(high_resolution_data, includes), batch_size=batch_size, shuffle=False)
    embeddings = DataLoader(np.array(e).mean(axis=1), batch_size=batch_size, shuffle=False)
    return low_resolution_dataloader, high_resolution_dataloader, embeddings

def upblock(in_channel, out_channel):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(True))

def downblock(in_channel, out_channel, with_batchnorm=True):
    if with_batchnorm:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True))

class Residual_Block(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.layer(x) + x
        return self.relu(out)

class C_Net(nn.Module):
    def __init__(self, embedding_size, mu_size) -> None:
        super().__init__()
        self.mu_size = mu_size
        self.layer = nn.Linear(embedding_size, 2*mu_size),

    def forward(self, embedding):
        o = self.layer(embedding)
        mu = o[:, :self.mu_size]
        sigma = o[:, self.mu_size:]
        std = sigma.mul(0.5).exp_()
        eps = Variable(torch.FloatTensor(std.size()).normal_().to(embedding.device))
        return eps.mul(std).add_(mu), mu, sigma
        