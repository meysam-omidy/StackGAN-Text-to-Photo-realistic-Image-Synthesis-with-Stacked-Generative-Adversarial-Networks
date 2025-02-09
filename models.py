import torch
from torch import nn
from utils import upblock, downblock, C_Net, Residual_Block


class Stage1_Generator(nn.Module):
    def __init__(self, embedding_size, mu_size, noise_size) -> None:
        super().__init__()
        self.noise_size = noise_size
        self.c_layer = C_Net(embedding_size, mu_size)
        ncm = noise_size + mu_size
        self.fc = nn.Sequential(
            nn.Linear(ncm, ncm*16, False),
            nn.BatchNorm1d(ncm*16),
            nn.ReLU(True)
        )
        self.up_layer = nn.Sequential(
            upblock(ncm, ncm//2),
            upblock(ncm//2, ncm//4),
            upblock(ncm//4, ncm//8),
            upblock(ncm//8, ncm//16),
        )
        self.last_layer = nn.Sequential(
            nn.Conv2d(ncm//16, 3, kernel_size=3, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, embedding, noise=None):
        batch_size = embedding.shape[0]
        if noise is None:
            noise = torch.randn(batch_size, self.noise_size).to(embedding.device)
        c0, m, s = self.c_layer(embedding)
        d = torch.concatenate([c0,noise], 1)
        d = self.fc(d).view(batch_size, -1, 4, 4)
        output = self.up_layer(d)
        output = self.last_layer(output)
        return output,m,s

class Stage1_Discriminator(nn.Module):
    def __init__(self, embedding_size, embedding_compress_size) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding_compress_size = embedding_compress_size
        self.ec = nn.Linear(embedding_size, embedding_compress_size)
        self.down_layer = nn.Sequential(
            downblock(3, 16, with_batchnorm=False),
            downblock(16, 64),
            downblock(64, 128),
            downblock(128, 512)
        )
        self.conv1 = nn.Conv2d(self.embedding_compress_size+512, 256, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Linear(4*4*256, 1),
            nn.Sigmoid()
        )

    def forward(self, image, embedding):
        batch_size = embedding.shape[0]
        embedding_compressed = self.ec(embedding)
        embedding_replicated = torch.randn(4, 4, batch_size, self.embedding_compress_size).to(embedding.device)
        embedding_replicated[:,:] = embedding_compressed
        embedding_replicated = embedding_replicated.permute(2,3,0,1)
        image_ = self.down_layer(image)
        output = torch.concatenate([image_, embedding_replicated], 1)
        output = self.conv1(output).view(batch_size, -1)
        output = self.fc(output)
        return output

class Stage2_Generator(nn.Module):
    def __init__(self, embedding_size, mu_size, base_gen) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.stage1_gen = base_gen
        for p in self.stage1_gen.parameters():
            p.requires_grad = False
        self.mu_size = mu_size
        self.c_layer = C_Net(embedding_size, mu_size)
        self.down_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            downblock(128, 256),
            downblock(256, 512)
        )
        self.conv_layer = nn.Sequential(
            nn.Conv2d(512+self.mu_size, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.residual_layer = nn.Sequential(
            Residual_Block(512),
            Residual_Block(512),
        )
        self.up_layer = nn.Sequential(
            upblock(512, 256),
            upblock(256, 128),
            upblock(128, 64),
            upblock(64, 3),
            nn.Sigmoid()
        )

    def forward(self, embedding, noise=None):
        batch_size = embedding.shape[0]
        if noise is None:
            noise = torch.randn(batch_size, 128).to(embedding.device)
        image,_,_ = self.stage1_gen(embedding)
        c0,m,s = self.c_layer(embedding)
        c0_replicated = torch.randn(16, 16, batch_size, self.mu_size).to(embedding.device)
        c0_replicated[:,:] = c0
        c0_replicated = c0_replicated.permute(2,3,0,1)
        image_ = self.down_layer(image)
        output = torch.concatenate([image_, c0_replicated], 1)
        output = self.conv_layer(output)
        output = self.residual_layer(output)
        output = self.up_layer(output)
        return output, image, m, s

class Stage2_Discriminator(nn.Module):
    def __init__(self, embedding_size, embedding_compress_size) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding_compress_size = embedding_compress_size
        self.ec = nn.Linear(embedding_size, embedding_compress_size)
        self.down_layer = nn.Sequential(
            downblock(3, 16, with_batchnorm=False),
            downblock(16, 32),
            downblock(32, 64),
            downblock(64, 128),
            downblock(128, 256),
            downblock(256, 512)
        )
        self.conv1 = nn.Conv2d(self.embedding_compress_size+512, 256, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Linear(4*4*256, 1),
            nn.Sigmoid()
        )

    def forward(self, image, embedding):
        batch_size = embedding.shape[0]
        embedding_compressed = self.ec(embedding)
        embedding_replicated = torch.randn(4, 4, batch_size, self.embedding_compress_size).to(embedding.device)
        embedding_replicated[:,:] = embedding_compressed
        embedding_replicated = embedding_replicated.permute(2,3,0,1)
        image_ = self.down_layer(image)
        output = torch.concatenate([image_, embedding_replicated], 1)
        output = self.conv1(output).view(batch_size, -1)
        output = self.fc(output)
        return output