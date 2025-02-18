import torch
import torch.nn as nn

class PConv(nn.Module):
    """Partial Convolution (PConv) вместо стандартной Conv"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(PConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones_like(x)

        x = self.conv(x * mask)
        mask = self.mask_conv(mask)
        mask = (mask > 0).float()
        x = self.bn(x)
        x = self.relu(x)

        return x, mask

class EMA(nn.Module):
    """Efficient Multi-scale Attention (EMA)"""
    def __init__(self, in_channels):
        super(EMA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv1(x)
        attention = self.conv2(attention)
        attention = self.bn(attention)
        attention = self.sigmoid(attention)
        return x * attention
