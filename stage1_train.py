"""Train Stage I of StackGAN on paired CUB images and caption embeddings."""

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.utils import save_image

from models import Stage1_Discriminator, Stage1_Generator
from utils import get_data


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    module.requires_grad_(enabled)


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())


def discriminator_hinge_loss(real_logits: torch.Tensor, wrong_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return (
        torch.relu(1 - real_logits).mean()
        + 0.5 * torch.relu(1 + wrong_logits).mean()
        + 0.5 * torch.relu(1 + fake_logits).mean()
    ) / 2


def generator_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


@torch.no_grad()
def update_ema(ema_model: nn.Module, source_model: nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(ema_model.parameters(), source_model.parameters()):
        ema_parameter.lerp_(parameter, 1 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), source_model.buffers()):
        ema_buffer.copy_(buffer)


def save_samples(generator: Stage1_Generator, embeddings: torch.Tensor, noise: torch.Tensor, path: Path) -> None:
    was_training = generator.training
    generator.eval()
    with torch.no_grad():
        images, _, _ = generator(embeddings, noise)
    save_image(images, path, nrow=4, normalize=True, value_range=(-1, 1))
    generator.train(was_training)


@torch.no_grad()
def validate(generator: Stage1_Generator, discriminator: Stage1_Discriminator, loader, kl_weight: float, device: torch.device, max_batches: int):
    """Return deterministic held-out hinge objectives without changing model state."""
    generator_was_training, discriminator_was_training = generator.training, discriminator.training
    generator.eval()
    discriminator.eval()
    total_d = total_g = total_kl = 0.0
    batches = 0
    # Keep validation sampling fixed while preserving the training RNG state.
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(2026)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(2026)
        for images, embeddings, _ in loader:
            images, embeddings = images.to(device, non_blocking=True), embeddings.to(device, non_blocking=True)
            wrong_embeddings = embeddings.roll(1, dims=0)
            fake_images, mu, logvar = generator(embeddings)
            d_loss = discriminator_hinge_loss(
                discriminator(images, embeddings),
                discriminator(images, wrong_embeddings),
                discriminator(fake_images, embeddings),
            )
            kl = kl_loss(mu, logvar)
            g_loss = generator_hinge_loss(discriminator(fake_images, embeddings)) + kl_weight * kl
            total_d += d_loss.item()
            total_g += g_loss.item()
            total_kl += kl.item()
            batches += 1
            if batches >= max_batches:
                break
    generator.train(generator_was_training)
    discriminator.train(discriminator_was_training)
    return total_d / batches, total_g / batches, total_kl / batches


def main(args: argparse.Namespace) -> None:
    if args.batch_size < 2:
        raise ValueError("batch-size must be at least 2 because mismatched caption pairs are used")
    if args.val_batches < 1:
        raise ValueError("val-batches must be at least 1")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_data(args.batch_size, image_size=64, num_workers=args.num_workers, bbox_padding=args.bbox_padding)
    validation_loader = get_data(min(args.batch_size, 16), image_size=64, split="test", training=False, num_workers=args.num_workers, bbox_padding=args.bbox_padding)
    _, fixed_embeddings, _ = next(iter(validation_loader))
    fixed_embeddings = fixed_embeddings.to(device)
    fixed_noise = torch.randn(len(fixed_embeddings), args.noise_size, device=device)

    generator = Stage1_Generator(noise_size=args.noise_size, base_channels=args.base_channels).to(device)
    discriminator = Stage1_Discriminator().to(device)
    generator_ema = copy.deepcopy(generator).eval()
    generator_ema.requires_grad_(False)
    optimizer_g = Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler_g = MultiStepLR(optimizer_g, milestones=[args.epochs // 2, args.epochs * 3 // 4], gamma=0.5)
    scheduler_d = MultiStepLR(optimizer_d, milestones=[args.epochs // 2, args.epochs * 3 // 4], gamma=0.5)

    output_dir = Path(args.output_dir)
    sample_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(exist_ok=True)
    print(f"Training Stage I on {len(train_loader.dataset)} pairs using {device}.")
    best_train_loss = float("inf")
    best_val_loss = float("inf")
    ema_updates = 0

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        total_d = total_g = total_kl = 0.0
        for images, embeddings, _ in train_loader:
            images, embeddings = images.to(device, non_blocking=True), embeddings.to(device, non_blocking=True)
            wrong_embeddings = embeddings.roll(1, dims=0)
            with torch.no_grad():
                fake_images, _, _ = generator(embeddings)
            d_loss = discriminator_hinge_loss(
                discriminator(images, embeddings),
                discriminator(images, wrong_embeddings),
                discriminator(fake_images, embeddings),
            )
            optimizer_d.zero_grad(set_to_none=True)
            d_loss.backward()
            optimizer_d.step()

            set_requires_grad(discriminator, False)
            # Keep the same discriminator behavior that it had during its own
            # update. It has no BatchNorm, so this is now state-free.
            fake_images, mu, logvar = generator(embeddings)
            kl = kl_loss(mu, logvar)
            g_loss = generator_hinge_loss(discriminator(fake_images, embeddings)) + args.kl_weight * kl
            optimizer_g.zero_grad(set_to_none=True)
            g_loss.backward()
            optimizer_g.step()
            ema_updates += 1
            ema_decay = min(args.ema_decay, (1 + ema_updates) / (10 + ema_updates))
            update_ema(generator_ema, generator, ema_decay)
            set_requires_grad(discriminator, True)

            total_d += d_loss.item()
            total_g += g_loss.item()
            total_kl += kl.item()

        scheduler_g.step()
        scheduler_d.step()
        batches = len(train_loader)
        train_d, train_g, train_kl = total_d / batches, total_g / batches, total_kl / batches
        val_d, val_g, val_kl = validate(generator_ema, discriminator, validation_loader, args.kl_weight, device, args.val_batches)
        train_loss, val_loss = train_d + train_g, val_d + val_g
        print(
            f"epoch {epoch:03d}/{args.epochs}: "
            f"train(d={train_d:.4f}, g={train_g:.4f}, kl={train_kl:.4f}) "
            f"val(d={val_d:.4f}, g={val_g:.4f}, kl={val_kl:.4f})"
        )
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            save_samples(generator_ema, fixed_embeddings, fixed_noise, sample_dir / f"epoch_{epoch:03d}.png")

        state = {"epoch": epoch, "generator": generator.state_dict(), "generator_ema": generator_ema.state_dict(), "discriminator": discriminator.state_dict(), "optimizer_g": optimizer_g.state_dict(), "optimizer_d": optimizer_d.state_dict(), "ema_updates": ema_updates, "train_loss": train_loss, "val_loss": val_loss, "args": vars(args)}
        torch.save(state, output_dir / "last.pt")
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(state, output_dir / "best_train.pt")
            print(f"  saved best training-loss checkpoint ({train_loss:.4f})")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(state, output_dir / "best_val.pt")
            print(f"  saved best validation-loss checkpoint ({val_loss:.4f})")
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            torch.save(state, output_dir / f"epoch_{epoch:03d}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--noise-size", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--kl-weight", type=float, default=2.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--bbox-padding", type=float, default=0.35, help="Context retained on each side of a CUB bird box")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=15)
    parser.add_argument("--val-batches", type=int, default=20, help="Held-out batches evaluated after each epoch")
    parser.add_argument("--output-dir", default="runs/stage1")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
