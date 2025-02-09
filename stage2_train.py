import torch
from torch import nn
from torch.optim import Adam
from models import Stage1_Generator, Stage2_Generator, Stage2_Discriminator
from utils import get_data

if __name__ == '__main__':
    BATCH_SIZE = 8
    N_EPOCHS = 300
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    low_resolution_dataloader, high_resolution_dataloader, embeddings = get_data(BATCH_SIZE)
    criterion = nn.BCELoss()
    stage1_gen = Stage1_Generator(1024, 128, 128).to(DEVICE)
    stage2_gen = Stage2_Generator(1024, 128, stage1_gen).to(DEVICE)
    stage2_disc = Stage2_Discriminator(1024,128).to(DEVICE)
    lr = 0.0002
    for epoch in range(N_EPOCHS):
        b = 0
        if epoch % 40 == 0:
            gen2_optimizer = Adam(stage2_gen.parameters(), lr)
            disc2_optimizer = Adam(stage2_disc.parameters(), lr)
            lr /= 2
        for embeds, (low_images,_), (high_images,_) in zip(embeddings, low_resolution_dataloader, high_resolution_dataloader):
            high_images, low_images, embeds = high_images.to(DEVICE), low_images.to(DEVICE), embeds.to(DEVICE)
            preds_real = stage2_disc(high_images, embeds)
            d_loss_real = criterion(preds_real, torch.ones_like(preds_real))
            fake_images,_,_,_ = stage2_gen(embeds)
            preds_fake = stage2_disc(fake_images, embeds)
            d_loss_fake = criterion(preds_fake, torch.zeros_like(preds_fake))
            d_loss = (d_loss_real + d_loss_fake) / 2
            disc2_optimizer.zero_grad()
            d_loss.backward()
            disc2_optimizer.step()
            
            for _ in range(1):
                fake_images,_,m,s = stage2_gen(embeds)
                preds_fake = stage2_disc(fake_images, embeds)
                kld = -0.5 * torch.mean(1 + s - m.pow(2) - s.exp())
                g_loss = criterion(preds_fake, torch.ones_like(preds_fake)) + kld
                gen2_optimizer.zero_grad()
                g_loss.backward()
                gen2_optimizer.step()