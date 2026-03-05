import torch
from torch import nn


class StudentBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class StudentNet(nn.Module):
    def __init__(self, in_channels, num_layers, max_out_channels, width=64, depth=3):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.max_out_channels = max_out_channels
        self.width = width
        self.depth = depth

        blocks = []
        c_in = in_channels + num_layers
        for i in range(depth):
            c_out = width
            blocks.append(StudentBlock(c_in, c_out))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(c_in, max_out_channels, kernel_size=1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    @staticmethod
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
