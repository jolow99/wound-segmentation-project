import torch
import sys
sys.path.append('../..')
from torch import nn
from torchvision.utils import save_image
import os
import datetime
from scripts.dataloader import DataGen, create_dataloaders
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter 

date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(f'runs/{date}')


def downsample(in_channels, out_channels, kernel_size, apply_batchnorm=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, bias=not apply_batchnorm)]
    if apply_batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*layers)

def upsample(in_channels, out_channels, kernel_size, apply_dropout=False):
    layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=1, bias=False)]
    layers.append(nn.BatchNorm2d(out_channels))
    if apply_dropout:
        layers.append(nn.Dropout(0.7))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down1 = downsample(3, 64, 4, apply_batchnorm=False)
        self.down2 = downsample(64, 128, 4)
        self.down3 = downsample(128, 256, 4)
        self.down4 = downsample(256, 512, 4)
        self.down5 = downsample(512, 512, 4)

        self.up1 = upsample(512, 512, 4, apply_dropout=True)
        self.up2 = upsample(1024, 256, 4, apply_dropout=True)  # 512 (up1 output) + 512 (d5) = 1024
        self.up3 = upsample(512, 128, 4)  # 512 (up2 output) + 512 (d4) = 1024, corrected to 1024 for symmetry
        self.up4 = upsample(256, 64, 4)  # 256 (up3 output) + 256 (d3) = 512, corrected to 512 for symmetry
        # self.up5 = upsample(128, 64, 4)  # 128 (up4 output) + 128 (d2) = , corrected to 256 for symmetry

        # Final layer corrected to take 128 channels as input: 64 (up5 output) + 64 (initial input channels) = 128
        self.last = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1),  # Corrected to ensure the output has 3 channels
            # nn.Tanh()
            nn.Sigmoid()
        )


    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5)
        u1 = torch.cat([u1, d4], dim=1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], dim=1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d2], dim=1)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, d1], dim=1)
        # u5 = self.up5(u4)
        final = self.last(u4)

        return self.last(u4)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.down1 = downsample(4, 64, 4, False)  # Input is concatenation of input image and target
        self.down2 = downsample(64, 128, 4)
        self.down3 = downsample(128, 256, 4)

        self.zero_pad1 = nn.ZeroPad2d(1)  # Padding to match the size
        self.conv = nn.Conv2d(256, 512, 4, stride=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(512)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.zero_pad2 = nn.ZeroPad2d(1)
        self.last = nn.Conv2d(512, 1, 4, stride=1)

    def forward(self, input, target):
        x = torch.cat([input, target], dim=1)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.zero_pad1(x)
        x = self.conv(x)
        x = self.batchnorm1(x)
        x = self.leaky_relu(x)
        x = self.zero_pad2(x)
        return self.last(x)


# Loss components
bce_loss = nn.BCEWithLogitsLoss()  # For the GAN loss
l1_loss = nn.L1Loss()  # For the L1 loss (image similarity)

def generator_loss(disc_fake_output, gen_output, target, lambda_l1=100):
    """
    disc_fake_output: Discriminator output for fake images
    gen_output: Generated images by the Generator
    target: Real images
    lambda_l1: Weight for L1 loss component
    """
    # GAN loss
    gan_loss = bce_loss(disc_fake_output, torch.ones_like(disc_fake_output))
    # L1 loss
    l1 = l1_loss(gen_output, target) * lambda_l1
    # Total generator loss
    total_gen_loss = gan_loss + l1
    return total_gen_loss, gan_loss, l1

def discriminator_loss(disc_real_output, disc_fake_output):
    """
    disc_real_output: Discriminator output for real images
    disc_fake_output: Discriminator output for fake images
    """
    # Loss for real images
    real_loss = bce_loss(disc_real_output, torch.ones_like(disc_real_output))
    # Loss for fake images
    fake_loss = bce_loss(disc_fake_output, torch.zeros_like(disc_fake_output))
    # Total discriminator loss
    total_disc_loss = real_loss + fake_loss
    return total_disc_loss

def train_step(generator, discriminator, input_image, target, optimizer_gen, optimizer_disc, lambda_l1=100):
    # Forward pass through the generator
    fake_image = generator(input_image)

    # Discriminator output for real and fake images
    disc_real_output = discriminator(input_image, target)
    disc_fake_output = discriminator(input_image, fake_image.detach())  # Detach to avoid computing gradients through the generator

    # Calculate discriminator loss and update
    optimizer_disc.zero_grad()
    disc_loss = discriminator_loss(disc_real_output, disc_fake_output)
    disc_loss.backward()
    optimizer_disc.step()

    # Second forward pass through the discriminator (for generator update)
    disc_fake_output = discriminator(input_image, fake_image)
    # Calculate generator loss and update
    optimizer_gen.zero_grad()
    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_fake_output, fake_image, target, lambda_l1)
    gen_total_loss.backward()
    optimizer_gen.step()

    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss


def fit(generator, discriminator, train_loader, valid_loader, optimizer_gen, optimizer_disc, epochs, device, checkpoint_dir='./training_checkpoints', lambda_l1=100):
    # Move models to the target device
    generator.to(device)
    discriminator.to(device)

    # For each epoch
    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        # Training loop
        for batch_idx, (input_image, target) in enumerate(train_loader):
            input_image, target = input_image.to(device), target.to(device)


            # Perform a training step
            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(
                generator, discriminator, input_image, target, optimizer_gen, optimizer_disc, lambda_l1
            )

            # Logging
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch {batch_idx}/{len(train_loader)}\n"
                      f"Loss D: {disc_loss.item()}, Loss G: {gen_total_loss.item()}, "
                      f"Loss G GAN: {gen_gan_loss.item()}, Loss G L1: {gen_l1_loss.item()}")

        # Validation loop (optional, for example, save generated images for visual inspection)
        if valid_loader is not None:
            generator.eval()
            with torch.no_grad():
                for input_image, target in valid_loader:
                    input_image, target = input_image.to(device), target.to(device)
                    fake_image = generator(input_image)
                    # Save or display images
                    save_image(fake_image, f"outputs/val_epoch_{epoch+1}.png", normalize=True)

        # Save checkpoints
        if checkpoint_dir:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            torch.save(generator.state_dict(), os.path.join(checkpoint_dir, f"generator_epoch_{epoch+1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, f"discriminator_epoch_{epoch+1}.pth"))

        writer.add_scalar('Loss/Generator', gen_total_loss.item(), epoch)
        writer.add_scalar('Loss/Discriminator', disc_loss.item(), epoch)


# dataloader = DataGen('/Users/jefflai/Desktop/ADL/wound-segmentation-project/data/wound_dataset/azh_wound_care_center_dataset_patches/',0.8, 224, 224, 'rgb')
# train_ds, valid_ds, test_ds = create_dataloaders(dataloader, 16, 'cpu')
# generator = Generator()
# discriminator = Discriminator()
# optimizer_gen = torch.optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
# optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))
# fit(generator, discriminator, train_ds, valid_ds, optimizer_gen, optimizer_disc, 20, 'cpu', f'./training_checkpoints/{date}', 100)


def generate_images(model, test_input, tar):
  prediction = model(test_input)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(np.transpose(display_list[i].detach().numpy(), (1, 2, 0)))
    plt.axis('off')
  plt.show()

# # load model 
# generator = Generator()
# generator.load_state_dict(torch.load('training_checkpoints/20240417-142253/generator_epoch_15.pth'))

# # #load test data
# test_input, tar = next(iter(test_ds))
# generate_images(generator, test_input,tar)

