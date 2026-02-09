"""
Network architectures for DRL trading agents.
- ResNet18 backbone (without last 2 layers) + CBAM attention
- DQN network
- Dueling DQN network
- PPO Actor-Critic networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    RESNET_FEATURE_DIM, STATE_DIM, NUM_ACTIONS,
    MLP_HIDDEN1, MLP_HIDDEN2, MLP_RAW_HIDDEN1, MLP_RAW_HIDDEN2,
    CBAM_REDUCTION, CBAM_KERNEL_SIZE, ILLEGAL_ACTION_VALUE, DEVICE
)


# ============================================================
# CBAM: Convolutional Block Attention Module
# ============================================================

class ChannelAttention(nn.Module):
    """Channel attention module of CBAM."""

    def __init__(self, channels: int, reduction: int = CBAM_REDUCTION):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial attention module of CBAM."""

    def __init__(self, kernel_size: int = CBAM_KERNEL_SIZE):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels: int, reduction: int = CBAM_REDUCTION,
                 kernel_size: int = CBAM_KERNEL_SIZE):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ============================================================
# ResNet18 + CBAM Backbone
# ============================================================

class ResNetCBAMBackbone(nn.Module):
    """
    ResNet18 backbone without last 2 layers + CBAM.
    Outputs a 512-dimensional feature vector.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)

        # Remove last 2 layers (avgpool and fc)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # Add CBAM after last conv layer
        self.cbam = CBAM(512)  # ResNet18 last conv has 512 channels

        # Global average pooling to get 512-d vector
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor [B, 3, 224, 224]
        Returns:
            Feature vector [B, 512]
        """
        features = self.features(x)
        features = self.cbam(features)
        features = self.avgpool(features)
        return features.view(features.size(0), -1)

    def get_attention_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get attention maps for visualization."""
        features = self.features(x)
        # Channel attention
        ca_out = self.cbam.channel_attention(features)
        # Spatial attention
        avg_out = torch.mean(ca_out, dim=1, keepdim=True)
        max_out, _ = torch.max(ca_out, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        spatial_map = self.cbam.spatial_attention.sigmoid(
            self.cbam.spatial_attention.conv(combined)
        )
        return features, spatial_map


# ============================================================
# DQN Network
# ============================================================

class DQNNetwork(nn.Module):
    """
    DQN network with ResNet18+CBAM backbone.
    State = [CNN features (512), funding_rate (1), position (1)] -> Q values (4)
    """

    def __init__(self, use_images: bool = True, raw_dim: int = 336):
        super().__init__()
        self.use_images = use_images

        if use_images:
            self.backbone = ResNetCBAMBackbone(pretrained=True)
            input_dim = STATE_DIM  # 514
            h1, h2 = MLP_HIDDEN1, MLP_HIDDEN2
        else:
            self.backbone = None
            input_dim = raw_dim + 2  # raw data + funding_rate + position
            h1, h2 = MLP_RAW_HIDDEN1, MLP_RAW_HIDDEN2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, NUM_ACTIONS),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        """
        Args:
            obs: dict with 'image' or 'raw_data', 'funding_rate', 'position'
        Returns:
            Q-values [B, 4]
        """
        if self.use_images:
            image = obs['image'].float() / 255.0  # Normalize to [0, 1]
            if image.dim() == 3:
                image = image.unsqueeze(0)
            # Permute from [B, H, W, C] to [B, C, H, W] if needed
            if image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)
            features = self.backbone(image)
        else:
            features = obs['raw_data']
            if features.dim() == 1:
                features = features.unsqueeze(0)

        funding = obs['funding_rate']
        position = obs['position']

        if funding.dim() == 1:
            funding = funding.unsqueeze(0)
        if position.dim() == 1:
            position = position.unsqueeze(0)

        state = torch.cat([features, funding, position], dim=-1)
        return self.mlp(state)


# ============================================================
# Dueling DQN Network
# ============================================================

class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN: separates state value and advantage streams.
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    """

    def __init__(self, use_images: bool = True, raw_dim: int = 336):
        super().__init__()
        self.use_images = use_images

        if use_images:
            self.backbone = ResNetCBAMBackbone(pretrained=True)
            input_dim = STATE_DIM
            h1, h2 = MLP_HIDDEN1, MLP_HIDDEN2
        else:
            self.backbone = None
            input_dim = raw_dim + 2
            h1, h2 = MLP_RAW_HIDDEN1, MLP_RAW_HIDDEN2

        # Shared feature layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, NUM_ACTIONS),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        if self.use_images:
            image = obs['image'].float() / 255.0
            if image.dim() == 3:
                image = image.unsqueeze(0)
            if image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)
            features = self.backbone(image)
        else:
            features = obs['raw_data']
            if features.dim() == 1:
                features = features.unsqueeze(0)

        funding = obs['funding_rate']
        position = obs['position']
        if funding.dim() == 1:
            funding = funding.unsqueeze(0)
        if position.dim() == 1:
            position = position.unsqueeze(0)

        state = torch.cat([features, funding, position], dim=-1)
        shared = self.shared(state)

        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)

        # Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


# ============================================================
# PPO Actor-Critic Networks
# ============================================================

class PPOActorNetwork(nn.Module):
    """PPO Actor: outputs action probabilities."""

    def __init__(self, use_images: bool = True, raw_dim: int = 336):
        super().__init__()
        self.use_images = use_images

        if use_images:
            self.backbone = ResNetCBAMBackbone(pretrained=True)
            input_dim = STATE_DIM
            h1, h2 = MLP_HIDDEN1, MLP_HIDDEN2
        else:
            self.backbone = None
            input_dim = raw_dim + 2
            h1, h2 = MLP_RAW_HIDDEN1, MLP_RAW_HIDDEN2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, NUM_ACTIONS),
        )

    def forward(self, obs: dict, action_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Returns action log-probabilities (after masking illegal actions).
        """
        if self.use_images:
            image = obs['image'].float() / 255.0
            if image.dim() == 3:
                image = image.unsqueeze(0)
            if image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)
            features = self.backbone(image)
        else:
            features = obs['raw_data']
            if features.dim() == 1:
                features = features.unsqueeze(0)

        funding = obs['funding_rate']
        position = obs['position']
        if funding.dim() == 1:
            funding = funding.unsqueeze(0)
        if position.dim() == 1:
            position = position.unsqueeze(0)

        state = torch.cat([features, funding, position], dim=-1)
        logits = self.mlp(state)

        # Apply action mask
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits + (1 - action_mask) * ILLEGAL_ACTION_VALUE

        return logits


class PPOCriticNetwork(nn.Module):
    """PPO Critic: outputs state value."""

    def __init__(self, use_images: bool = True, raw_dim: int = 336):
        super().__init__()
        self.use_images = use_images

        if use_images:
            self.backbone = ResNetCBAMBackbone(pretrained=True)
            input_dim = STATE_DIM
            h1, h2 = MLP_HIDDEN1, MLP_HIDDEN2
        else:
            self.backbone = None
            input_dim = raw_dim + 2
            h1, h2 = MLP_RAW_HIDDEN1, MLP_RAW_HIDDEN2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        if self.use_images:
            image = obs['image'].float() / 255.0
            if image.dim() == 3:
                image = image.unsqueeze(0)
            if image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)
            features = self.backbone(image)
        else:
            features = obs['raw_data']
            if features.dim() == 1:
                features = features.unsqueeze(0)

        funding = obs['funding_rate']
        position = obs['position']
        if funding.dim() == 1:
            funding = funding.unsqueeze(0)
        if position.dim() == 1:
            position = position.unsqueeze(0)

        state = torch.cat([features, funding, position], dim=-1)
        return self.mlp(state)
