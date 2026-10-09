import torch
from torch import nn
from torch.nn.utils import spectral_norm

from utils import C_Net, Residual_Block, downblock, upblock


class Stage1_Generator(nn.Module):
    """Generate a 64x64 image from a caption embedding."""

    def __init__(self, embedding_size: int = 1024, condition_size: int = 128, noise_size: int = 100, base_channels: int = 64) -> None:
        super().__init__()
        self.noise_size = noise_size
        self.c_layer = C_Net(embedding_size, condition_size)
        self.fc = nn.Sequential(
            nn.Linear(condition_size + noise_size, 4 * 4 * base_channels * 8, bias=False),
            nn.BatchNorm1d(4 * 4 * base_channels * 8),
            nn.ReLU(inplace=True),
        )
        self.up_layer = nn.Sequential(
            upblock(base_channels * 8, base_channels * 4),
            upblock(base_channels * 4, base_channels * 2),
            upblock(base_channels * 2, base_channels),
            upblock(base_channels, base_channels // 2),
        )
        self.last_layer = nn.Sequential(nn.Conv2d(base_channels // 2, 3, 3, padding=1, bias=False), nn.Tanh())

    def forward(self, embedding: torch.Tensor, noise: torch.Tensor | None = None):
        if noise is None:
            noise = torch.randn(embedding.shape[0], self.noise_size, device=embedding.device)
        condition, mu, logvar = self.c_layer(embedding)
        image = self.fc(torch.cat((condition, noise), dim=1)).view(embedding.shape[0], -1, 4, 4)
        return self.last_layer(self.up_layer(image)), mu, logvar


class _ConditionalDiscriminator(nn.Module):
    def __init__(self, embedding_size: int, condition_size: int, image_size: int) -> None:
        super().__init__()
        channels = [3, 64, 128, 256, 512]
        # Discriminators deliberately have no BatchNorm: batch-dependent
        # statistics made the old train/eval paths disagree and enabled collapse.
        blocks = [downblock(channels[0], channels[1], with_batchnorm=False, spectral=True)]
        blocks += [downblock(in_channels, out_channels, with_batchnorm=False, spectral=True) for in_channels, out_channels in zip(channels[1:], channels[2:])]
        while image_size // (2 ** len(blocks)) > 4:
            blocks.append(downblock(512, 512, with_batchnorm=False, spectral=True))
        self.down_layer = nn.Sequential(*blocks)
        self.condition = spectral_norm(nn.Linear(embedding_size, condition_size))
        self.joint = nn.Sequential(
            spectral_norm(nn.Conv2d(512 + condition_size, 512, 3, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, 4, bias=False)),
        )

    def forward(self, image: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        features = self.down_layer(image)
        condition = self.condition(embedding).unsqueeze(-1).unsqueeze(-1)
        condition = condition.expand(-1, -1, features.shape[2], features.shape[3])
        return self.joint(torch.cat((features, condition), dim=1)).flatten(1)


class Stage1_Discriminator(_ConditionalDiscriminator):
    def __init__(self, embedding_size: int = 1024, condition_size: int = 128) -> None:
        super().__init__(embedding_size, condition_size, image_size=64)


class Stage2_Generator(nn.Module):
    """Refine a frozen Stage-I image into a 256x256 image."""

    def __init__(self, embedding_size: int, condition_size: int, base_gen: Stage1_Generator, base_channels: int = 64) -> None:
        super().__init__()
        self.stage1_gen = base_gen
        self.stage1_gen.requires_grad_(False)
        self.stage1_gen.eval()
        self.c_layer = C_Net(embedding_size, condition_size)
        self.down_layer = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1, bias=False), nn.ReLU(inplace=True),
            downblock(base_channels * 4, base_channels * 8), downblock(base_channels * 8, base_channels * 8),
        )
        self.joint = nn.Sequential(
            nn.Conv2d(base_channels * 8 + condition_size, base_channels * 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8), nn.ReLU(inplace=True),
        )
        self.residual_layer = nn.Sequential(Residual_Block(base_channels * 8), Residual_Block(base_channels * 8))
        self.up_layer = nn.Sequential(
            upblock(base_channels * 8, base_channels * 4), upblock(base_channels * 4, base_channels * 2),
            upblock(base_channels * 2, base_channels), upblock(base_channels, base_channels // 2),
        )
        self.last_layer = nn.Sequential(nn.Conv2d(base_channels // 2, 3, 3, padding=1, bias=False), nn.Tanh())

    def train(self, mode: bool = True):
        super().train(mode)
        self.stage1_gen.eval()
        return self

    def forward(self, embedding: torch.Tensor, noise: torch.Tensor | None = None):
        del noise  # Stage II randomness is supplied by conditioning augmentation.
        self.stage1_gen.eval()
        with torch.no_grad():
            stage1_image, _, _ = self.stage1_gen(embedding)
        condition, mu, logvar = self.c_layer(embedding)
        features = self.down_layer(stage1_image)
        condition = condition.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, features.shape[2], features.shape[3])
        image = self.residual_layer(self.joint(torch.cat((features, condition), dim=1)))
        return self.last_layer(self.up_layer(image)), stage1_image, mu, logvar


class Stage2_Discriminator(nn.Module):
    """Three conditional PatchGANs supervising 64, 128, and 256 pixel detail."""

    def __init__(self, embedding_size: int = 1024, condition_size: int = 128, scales: tuple[int, ...] = (64, 128, 256)) -> None:
        super().__init__()
        self.scales = scales
        self.discriminators = nn.ModuleDict(
            {
                str(scale): _ConditionalPatchDiscriminator(embedding_size, condition_size, image_size=scale)
                for scale in scales
            }
        )

    def forward(self, image: torch.Tensor, embedding: torch.Tensor, return_features: bool = False):
        logits, features = [], []
        for scale, discriminator in self.discriminators.items():
            scale_size = int(scale)
            scaled_image = image if image.shape[-1] == scale_size else nn.functional.interpolate(
                image,
                size=(scale_size, scale_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            patch_logits, patch_features = discriminator(scaled_image, embedding)
            logits.append(patch_logits)
            features.append(patch_features)
        return (logits, features) if return_features else logits


class _ConditionalPatchDiscriminator(nn.Module):
    """A BatchNorm-free conditional discriminator returning a 4x4 patch map."""

    def __init__(self, embedding_size: int, condition_size: int, image_size: int) -> None:
        super().__init__()
        channels = [3, 64, 128, 256, 512]
        blocks = [downblock(channels[0], channels[1], with_batchnorm=False, spectral=True)]
        blocks += [
            downblock(in_channels, out_channels, with_batchnorm=False, spectral=True)
            for in_channels, out_channels in zip(channels[1:], channels[2:])
        ]
        while image_size // (2 ** len(blocks)) > 4:
            blocks.append(downblock(512, 512, with_batchnorm=False, spectral=True))
        self.down_layer = nn.Sequential(*blocks)
        self.condition = spectral_norm(nn.Linear(embedding_size, condition_size))
        self.joint = nn.Sequential(
            spectral_norm(nn.Conv2d(512 + condition_size, 512, 3, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.patch_head = spectral_norm(nn.Conv2d(512, 1, 3, padding=1, bias=False))

    def forward(self, image: torch.Tensor, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.down_layer(image)
        condition = self.condition(embedding).unsqueeze(-1).unsqueeze(-1)
        condition = condition.expand(-1, -1, image_features.shape[2], image_features.shape[3])
        features = self.joint(torch.cat((image_features, condition), dim=1))
        return self.patch_head(features), features
