import torch
import torch.nn as nn

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, dropout=0.0):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if batch_norm:
            layers.insert(1, nn.BatchNorm2d(out_channels))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)
    
class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()
        n_channels = config.in_channels
        n_classes = config.n_classes
        depth = config.depth
        n_filters = config.n_filters
        batch_norm = config.batch_norm
        dropout = config.dropout
        
        self.inc = DoubleConv(n_channels, n_filters, batch_norm=batch_norm, dropout=dropout)


        self.down_convs = nn.ModuleList()
        for i in range(depth - 1):
            self.down_convs.append(
                nn.Sequential(
                    nn.MaxPool2d(2),
                    DoubleConv(n_filters * (2 ** i), n_filters * (2 ** (i + 1)), batch_norm=batch_norm, dropout=dropout)
                )
            )

        self.up_convs = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.up_convs.append(
                nn.ConvTranspose2d(n_filters * (2 ** i), n_filters * (2 ** (i - 1)), kernel_size=2, stride=2)
            )
            self.up_convs.append(
                DoubleConv(n_filters * (2 ** (i - 1)) * 2, n_filters * (2 ** (i - 1)), batch_norm=batch_norm, dropout=dropout)
            )

        self.outc = nn.Conv2d(n_filters, n_classes, kernel_size=1)

    def forward(self, x):
        x_skip_connections = []

        x = self.inc(x)
        for down_conv in self.down_convs:
            x_skip_connections.append(x)
            x = down_conv(x)

        for i in range(0, len(self.up_convs), 2):
            x = self.up_convs[i](x)
            x = torch.cat([x, x_skip_connections.pop()], dim=1)
            x = self.up_convs[i + 1](x)

        x = self.outc(x)
        return x 
