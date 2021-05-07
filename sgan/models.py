import torch
import torch.nn as nn


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class BasicGenerator(nn.Module):
    def __init__(self):
        super(BasicGenerator, self).__init__()
        # input is 100x1
        # output is 1x224x224
        self.block0 = nn.Sequential(
            # project and reshape to 14x14
            nn.Linear(100, 1024 * 14 * 14),
            nn.LeakyReLU(0.2),
        )
        self.block1 = nn.Sequential(
            # upsample to 28x28
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # upsample to 56x56
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # upsample to 112x112
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # upsample to 224x224
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # convert to 1 channel and add sigmoid so values are between 0-1
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        b_size = x.shape[0]
        x = self.block0(x)
        x = torch.reshape(x, (b_size, 1024, 14, 14))
        return self.block1(x)


class BasicDiscriminator(nn.Module):
    def __init__(self, k=6):
        super(BasicDiscriminator, self).__init__()
        # input is 1x224x224
        # output is 1x(K+1)
        self.block = nn.Sequential(
            # downsample to 112x112
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # downsample to 56x56
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # downsample to 28x28
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # downsample to 14x14
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            # convert to 1 channel
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, k + 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.block(input)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def generate_noise(b_size):
    # Generates a latent vector of gaussian sampled random values
    return torch.randn(b_size, 100)

# Testing output sizes with random input
# g_net = BasicGenerator()
# g_net.apply(weights_init)
# test_input = generate_noise(10)
# print(test_input.shape)
# out = g_net(test_input)
# print(out.shape)
#
# d_net = BasicDiscriminator()
# d_net.apply(weights_init)
# test_input = torch.rand((1, 1, 224, 224))
# out = d_net(test_input)
# print(out.shape)
