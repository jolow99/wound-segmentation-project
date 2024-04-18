import torch
import torch.nn as nn

class SegNet(nn.Module):
    def __init__(self, config):
        super(SegNet, self).__init__()
        n_filters = config.n_filters
        self.input_dim_x = config.input_dim_x
        self.input_dim_y = config.input_dim_y
        num_channels = config.num_channels

        self.encoder_conv1 = nn.Conv2d(num_channels, n_filters, kernel_size=9, padding=4)
        self.encoder_conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=5, padding=2)
        self.encoder_conv3 = nn.Conv2d(n_filters, 2 * n_filters, kernel_size=5, padding=2)
        self.encoder_conv4 = nn.Conv2d(2 * n_filters, 2 * n_filters, kernel_size=5, padding=2)

        self.conv5 = nn.Conv2d(2 * n_filters, n_filters, kernel_size=5, padding=2)

        self.decoder_conv6 = nn.Conv2d(n_filters, n_filters, kernel_size=7, padding=3)
        self.decoder_conv7 = nn.Conv2d(n_filters, n_filters, kernel_size=5, padding=2)
        self.decoder_conv8 = nn.Conv2d(n_filters, n_filters, kernel_size=5, padding=2)
        self.decoder_conv9 = nn.Conv2d(n_filters, 1, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        encoder_conv1 = self.relu(self.encoder_conv1(x))
        pool1 = self.pool(encoder_conv1)
        encoder_conv2 = self.relu(self.encoder_conv2(pool1))
        pool2 = self.pool(encoder_conv2)
        encoder_conv3 = self.relu(self.encoder_conv3(pool2))
        pool3 = self.pool(encoder_conv3)
        encoder_conv4 = self.relu(self.encoder_conv4(pool3))
        pool4 = self.pool(encoder_conv4)

        # Bridge
        conv5 = self.relu(self.conv5(pool4))

        # Decoder
        decoder_conv6 = self.relu(self.decoder_conv6(self.upsample(conv5)))
        decoder_conv7 = self.relu(self.decoder_conv7(self.upsample(decoder_conv6)))
        decoder_conv8 = self.relu(self.decoder_conv8(self.upsample(decoder_conv7)))
        decoder_conv9 = self.decoder_conv9(self.upsample(decoder_conv8))

        return decoder_conv9