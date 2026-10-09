"""Train Stage II after Stage I has been trained with stage1_train.py."""

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.utils import save_image

from models import Stage1_Generator, Stage2_Discriminator, Stage2_Generator
from utils import get_data


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    module.requires_grad_(enabled)


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())


def discriminator_hinge_loss(real_logits: list[torch.Tensor], wrong_logits: list[torch.Tensor], fake_logits: list[torch.Tensor]) -> torch.Tensor:
    """Average the conditional hinge objective over all image scales."""
    losses = [
        (
            torch.relu(1 - real_scale).mean()
            + 0.5 * torch.relu(1 + wrong_scale).mean()
            + 0.5 * torch.relu(1 + fake_scale).mean()
        ) / 2
        for real_scale, wrong_scale, fake_scale in zip(real_logits, wrong_logits, fake_logits)
    ]
    return torch.stack(losses).mean()


def generator_hinge_loss(fake_logits: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([-logits.mean() for logits in fake_logits]).mean()


def feature_matching_loss(real_features: list[torch.Tensor], fake_features: list[torch.Tensor]) -> torch.Tensor:
    """Match mean discriminator features at every scale for stable local detail."""
    losses = [
        F.l1_loss(fake.mean(dim=0), real.detach().mean(dim=0))
        for real, fake in zip(real_features, fake_features)
    ]
    return torch.stack(losses).mean()


@torch.no_grad()
def update_ema(ema_model: nn.Module, source_model: nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(ema_model.parameters(), source_model.parameters()):
        ema_parameter.lerp_(parameter, 1 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), source_model.buffers()):
        ema_buffer.copy_(buffer)


def load_stage1(path: str, device: torch.device, noise_size: int, base_channels: int) -> Stage1_Generator:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("generator_ema", checkpoint.get("generator")) if isinstance(checkpoint, dict) else checkpoint
    model = Stage1_Generator(noise_size=noise_size, base_channels=base_channels).to(device)
    model.load_state_dict(state_dict)
    return model


def save_samples(generator: Stage2_Generator, embeddings: torch.Tensor, path: Path) -> None:
    was_training = generator.training
    generator.eval()
    with torch.no_grad():
        images, stage1_images, _, _ = generator(embeddings)
    stage1_images = F.interpolate(stage1_images, size=images.shape[-2:], mode="bilinear", align_corners=False)
    save_image(torch.cat((stage1_images, images), dim=0), path, nrow=len(images), normalize=True, value_range=(-1, 1))
    generator.train(was_training)


@torch.no_grad()
def validate(generator: Stage2_Generator, discriminator: Stage2_Discriminator, loader, kl_weight: float, feature_matching_weight: float, device: torch.device, max_batches: int):
    """Return deterministic held-out hinge objectives without changing model state."""
    generator_was_training, discriminator_was_training = generator.training, discriminator.training
    generator.eval()
    discriminator.eval()
    total_d = total_g = total_kl = total_fm = 0.0
    batches = 0
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(2026)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(2026)
        for images, embeddings, _ in loader:
            images, embeddings = images.to(device, non_blocking=True), embeddings.to(device, non_blocking=True)
            wrong_embeddings = embeddings.roll(1, dims=0)
            fake_images, _, mu, logvar = generator(embeddings)
            real_logits, real_features = discriminator(images, embeddings, return_features=True)
            fake_logits, fake_features = discriminator(fake_images, embeddings, return_features=True)
            d_loss = discriminator_hinge_loss(
                real_logits,
                discriminator(images, wrong_embeddings),
                fake_logits,
            )
            kl = kl_loss(mu, logvar)
            fm = feature_matching_loss(real_features, fake_features)
            g_loss = generator_hinge_loss(fake_logits) + kl_weight * kl + feature_matching_weight * fm
            total_d += d_loss.item()
            total_g += g_loss.item()
            total_kl += kl.item()
            total_fm += fm.item()
            batches += 1
            if batches >= max_batches:
                break
    generator.train(generator_was_training)
    discriminator.train(discriminator_was_training)
    return total_d / batches, total_g / batches, total_kl / batches, total_fm / batches


def main(args: argparse.Namespace) -> None:
    if args.batch_size < 2:
        raise ValueError("batch-size must be at least 2 because mismatched caption pairs are used")
    if args.val_batches < 1:
        raise ValueError("val-batches must be at least 1")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_data(args.batch_size, image_size=256, num_workers=args.num_workers, bbox_padding=args.bbox_padding)
    validation_loader = get_data(min(args.batch_size, 8), image_size=256, split="test", training=False, num_workers=args.num_workers, bbox_padding=args.bbox_padding)
    _, fixed_embeddings, _ = next(iter(validation_loader))
    fixed_embeddings = fixed_embeddings.to(device)

    stage1 = load_stage1(args.stage1_checkpoint, device, args.noise_size, args.base_channels)
    generator = Stage2_Generator(1024, 128, stage1, base_channels=args.base_channels).to(device)
    discriminator = Stage2_Discriminator().to(device)
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
    print(f"Training Stage II on {len(train_loader.dataset)} pairs using frozen Stage I on {device}.")
    best_train_loss = float("inf")
    best_val_loss = float("inf")
    ema_updates = 0

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        total_d = total_g = total_kl = total_fm = 0.0
        for images, embeddings, _ in train_loader:
            images, embeddings = images.to(device, non_blocking=True), embeddings.to(device, non_blocking=True)
            wrong_embeddings = embeddings.roll(1, dims=0)
            with torch.no_grad():
                fake_images, _, _, _ = generator(embeddings)
            d_loss = discriminator_hinge_loss(
                discriminator(images, embeddings),
                discriminator(images, wrong_embeddings),
                discriminator(fake_images, embeddings),
            )
            optimizer_d.zero_grad(set_to_none=True)
            d_loss.backward()
            optimizer_d.step()

            set_requires_grad(discriminator, False)
            fake_images, _, mu, logvar = generator(embeddings)
            with torch.no_grad():
                _, real_features = discriminator(images, embeddings, return_features=True)
            fake_logits, fake_features = discriminator(fake_images, embeddings, return_features=True)
            kl = kl_loss(mu, logvar)
            fm = feature_matching_loss(real_features, fake_features)
            g_loss = generator_hinge_loss(fake_logits) + args.kl_weight * kl + args.feature_matching_weight * fm
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
            total_fm += fm.item()

        scheduler_g.step()
        scheduler_d.step()
        batches = len(train_loader)
        train_d, train_g, train_kl, train_fm = total_d / batches, total_g / batches, total_kl / batches, total_fm / batches
        val_d, val_g, val_kl, val_fm = validate(generator_ema, discriminator, validation_loader, args.kl_weight, args.feature_matching_weight, device, args.val_batches)
        train_loss, val_loss = train_d + train_g, val_d + val_g
        print(
            f"epoch {epoch:03d}/{args.epochs}: "
            f"train(d={train_d:.4f}, g={train_g:.4f}, kl={train_kl:.4f}, fm={train_fm:.4f}) "
            f"val(d={val_d:.4f}, g={val_g:.4f}, kl={val_kl:.4f}, fm={val_fm:.4f})"
        )
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            save_samples(generator_ema, fixed_embeddings, sample_dir / f"epoch_{epoch:03d}.png")

        state = {"epoch": epoch, "generator": generator.state_dict(), "generator_ema": generator_ema.state_dict(), "discriminator": discriminator.state_dict(), "ema_updates": ema_updates, "train_loss": train_loss, "val_loss": val_loss, "feature_matching_loss": val_fm, "args": vars(args)}
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
    parser.add_argument("--stage1-checkpoint", required=True, help="Checkpoint produced by stage1_train.py")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--noise-size", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--kl-weight", type=float, default=2.0)
    parser.add_argument("--feature-matching-weight", type=float, default=10.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--bbox-padding", type=float, default=0.35, help="Context retained on each side of a CUB bird box")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=15)
    parser.add_argument("--val-batches", type=int, default=20, help="Held-out batches evaluated after each epoch")
    parser.add_argument("--output-dir", default="runs/stage2")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
