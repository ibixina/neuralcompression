import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x)
        return x


class TeacherNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=100, channels=None):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 512]
        self.channels = channels

        blocks = []
        c_in = in_channels
        for i, c_out in enumerate(channels):
            pool = True if i < len(channels) - 1 else False
            blocks.append(ConvBlock(c_in, c_out, pool=pool))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c_in, num_classes),
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)

    @torch.no_grad()
    def forward_with_activations(self, x):
        layer_io = []
        for block in self.blocks:
            x_in = x
            x = block(x)
            layer_io.append((x_in, x))
        logits = self.classifier(x)
        return logits, layer_io
