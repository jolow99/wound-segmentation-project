import torch.nn as nn 

class Autoencoder(nn.Module):
    def __init__(self, config):
        super(Autoencoder, self).__init__()
        in_channels = config.in_channels
        out_channels = config.out_channels
        n_filters = config.n_filters

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(n_filters, n_filters*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters*2, n_filters*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(n_filters*2, n_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_filters, n_filters, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(n_filters, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder(x)

        # Middle
        x2 = self.middle(x1)

        # Decoder
        x3 = self.decoder(x2)

        return x3
