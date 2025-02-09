import torch
from torch import nn
from torch.optim import Adam
from models import Stage1_Generator, Stage1_Discriminator
from utils import get_data

if __name__ == '__main__':
    BATCH_SIZE = 8
    N_EPOCHS = 300
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    low_resolution_dataloader, high_resolution_dataloader, embeddings = get_data(BATCH_SIZE)
    stage1_gen = Stage1_Generator(1024, 128, 128).to(DEVICE)
    stage1_disc = Stage1_Discriminator(1024,128).to(DEVICE)
    criterion = nn.BCELoss()
    lr = 0.0002
    gen1_optimizer = Adam(stage1_gen.parameters(), lr)
    disc1_optimizer = Adam(stage1_disc.parameters(), lr)
    for epoch in range(N_EPOCHS):
        if epoch % 100 == 0:
            gen1_optimizer = Adam(stage1_gen.parameters(), lr)
            disc1_optimizer = Adam(stage1_disc.parameters(), lr)
            lr /= 2
        for embeds, (images,_) in zip(embeddings, low_resolution_dataloader):
            images, embeds = images.to(DEVICE), embeds.to(DEVICE)
            preds_real = stage1_disc(images, embeds)
            d_loss_real = criterion(preds_real, torch.ones_like(preds_real))
            fake_images,_,_ = stage1_gen(embeds)
            preds_fake = stage1_disc(fake_images, embeds)
            d_loss_fake = criterion(preds_fake, torch.zeros_like(preds_fake))
            d_loss = (d_loss_real + d_loss_fake) / 2
            disc1_optimizer.zero_grad()
            d_loss.backward()
            disc1_optimizer.step()

            for _ in range(5):
                fake_images,m,s = stage1_gen(embeds)
                preds_fake = stage1_disc(fake_images, embeds)
                kld = -0.5 * torch.mean(1 + s - m.pow(2) - s.exp())
                g_loss = criterion(preds_fake, torch.ones_like(preds_fake)) + kld
                gen1_optimizer.zero_grad()
                g_loss.backward()
                gen1_optimizer.step()