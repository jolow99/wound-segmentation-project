import torch.nn as nn
import torch

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
    
class MoEBlock(nn.Module):
    def __init__(self, config):
        super(MoEBlock, self).__init__()
        self.moe = nn.ModuleList([
            nn.Conv2d(2048, 128, kernel_size=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(25088, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        ])
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        for layer in self.moe:
            x = layer(x)
        # x= nn.Sigmoid()(x)
        # x = self.moe(x)
        x = torch.softmax(x, dim=-1)
        return x

class MixCircleNet(nn.Module):
  def __init__(self, config):
    super(MixCircleNet, self).__init__()
    n_channels = config.in_channels
    n_classes = config.n_classes
    depth = config.depth
    n_filters = config.n_filters
    batch_norm = config.batch_norm
    dropout = config.dropout
    self.inc = DoubleConv(n_channels, n_filters, batch_norm=batch_norm, dropout=dropout)
    self.inc2 = DoubleConv(n_channels, n_filters, batch_norm=batch_norm, dropout=dropout)

    self.down_convs_expert1 = nn.ModuleList()
    self.down_convs_expert2 = nn.ModuleList()
    for i in range(depth - 1):
        self.down_convs_expert1.append(
            nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(n_filters * (2 ** i), n_filters * (2 ** (i + 1)), batch_norm=batch_norm, dropout=dropout)
            )
        )
    
    for i in range(depth - 1):
        self.down_convs_expert2.append(
            nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(n_filters * (2 ** i), n_filters * (2 ** (i + 1)), batch_norm=batch_norm, dropout=dropout)
            )
        )

    self.up_convs_expert1 = nn.ModuleList()
    for i in range(depth - 1, 0, -1):
        self.up_convs_expert1.append(
            nn.ConvTranspose2d(n_filters * (2 ** i), n_filters * (2 ** (i - 1)), kernel_size=2, stride=2)
        )
        self.up_convs_expert1.append(
            DoubleConv(n_filters * (2 ** (i - 1)) * 2, n_filters * (2 ** (i - 1)), batch_norm=batch_norm, dropout=dropout)
        )

    self.up_convs_expert2 = nn.ModuleList()
    for i in range(depth - 1, 0, -1):
        self.up_convs_expert2.append(
            nn.ConvTranspose2d(n_filters * (2 ** i), n_filters * (2 ** (i - 1)), kernel_size=2, stride=2)
        )
        self.up_convs_expert2.append(
            DoubleConv(n_filters * (2 ** (i - 1)) * 2, n_filters * (2 ** (i - 1)), batch_norm=batch_norm, dropout=dropout)
        )

    self.expert1_outc = nn.Conv2d(n_filters, n_classes, kernel_size=1)
    self.expert2_outc = nn.Conv2d(n_filters, n_classes, kernel_size=1)
    self.final_conv = nn.Conv2d(config.n_classes * 2, config.n_classes, kernel_size=1)  # Adjust based on your n_classes
    self.moe = MoEBlock(config)
    self.ensemble_conv = nn.Conv2d(config.n_classes, config.n_classes, kernel_size=3, padding="same")

  def forward(self, x):
    expert1_skip_connections = []
    expert2_skip_connections = []

    x1 = self.inc(x)
    x2 = self.inc2(x)

    for down_conv in self.down_convs_expert1:
        expert1_skip_connections.append(x1)
        x1 = down_conv(x1)
    
    for down_conv in self.down_convs_expert2:
        expert2_skip_connections.append(x2)
        x2 = down_conv(x2)


    weights = self.moe(x1, x2)

    for i in range(0, len(self.up_convs_expert1), 2):
        x1 = self.up_convs_expert1[i](x1)
        x1 = torch.cat([x1, expert1_skip_connections.pop()], dim=1)
        x1 = self.up_convs_expert1[i + 1](x1)

    for i in range(0, len(self.up_convs_expert2), 2):
        x2 = self.up_convs_expert2[i](x2)
        x2 = torch.cat([x2, expert2_skip_connections.pop()], dim=1)
        x2 = self.up_convs_expert2[i + 1](x2)

    expert1_out = self.expert1_outc(x1)
    expert2_out = self.expert2_outc(x2)

    out = expert1_out * weights[:, 0].reshape(-1,1,1,1) + expert2_out * weights[:, 1].reshape(-1,1,1,1)
    out = self.ensemble_conv(out)
    # out = torch.cat([expert1_out * weights[:, 0].reshape(-1,1,1,1), expert2_out * weights[:, 1].reshape(-1,1,1,1)], dim=1)
    # out = self.ensemble_conv(out)

    # ensemble_pred = torch.cat([pred1, pred2], dim=1)
    # ensemble_pred = self.final_conv(ensemble_pred)

    return out

# class EnsembleUNet(nn.Module):
#     def __init__(self, config):
#         super(EnsembleUNet, self).__init__()
#         self.unet1 = UNet(config)
#         self.unet2 = UNet(config)
#         # self.final_conv = nn.Conv2d(config.n_classes * 2, config.n_classes, kernel_size=1)
#         self.linear = nn.Linear(100352, 1*224*224) 

#     def forward(self, x):
#         pred1 = self.unet1(x)
#         pred2 = self.unet2(x)
#         ensemble_pred = torch.cat([pred1, pred2], dim=1)
#         print(ensemble_pred.size())
#         flat = nn.Flatten()(ensemble_pred)  # Flatten the output
#         print(flat.size())
#         print(self.linear)
#         out = self.linear(flat)
#         print(out.size())
#         # out = self.final_conv(ensemble_pred)
#         # ensemble_pred = self.linear(ensemble_pred)
#         # out = einops.rearrange(ensemble_pred, 'b (c h w) -> b c h w', c=1, h=224, w=224)

#         return out
