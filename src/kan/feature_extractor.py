import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *
from torch.autograd import Function


class BasicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)

        return out

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class EnhancedFeatureExtractor(nn.Module):
    def __init__(self,
        colors = 3):
        super(EnhancedFeatureExtractor, self).__init__()
        self.initial_layers = nn.Sequential(
            nn.Conv2d(colors, 32, kernel_size=3, stride=1, padding=1),  # Increased number of filters
            nn.ReLU(),
            nn.BatchNorm2d(32),  # Added Batch Normalization
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),  # Added Dropout
            BasicResBlock(32, 64),
            SEBlock(64, reduction=16),  # Squeeze-and-Excitation block
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),  # Added Dropout
            DepthwiseSeparableConv(64, 128, kernel_size=3),  # Increased number of filters
            nn.ReLU(),
            BasicResBlock(128, 256),
            SEBlock(256, reduction=16),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),  # Added Dropout
            SelfAttention(256),  # Added Self-Attention layer
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output for fully connected layers
        return x