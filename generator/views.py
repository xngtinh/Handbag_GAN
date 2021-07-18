from django.shortcuts import render
import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from keras.models import Sequential, load_model,Model

batch_size = 32
lr = 0.001
beta1 = 0.5
epochs = 500

real_label = 0.5
fake_label = 0
nz = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")
# device = torch.load(('./generator/Weights/discriminator_27_3.h5'),map_location=torch.device('cpu'))


class Generator(nn.Module):
    def __init__(self, nz=128, channels=3):
        super(Generator, self).__init__()

        self.nz = nz
        self.channels = channels

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0):
            block = [
                nn.ConvTranspose2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(n_output),
                nn.ReLU(inplace=True),
            ]
            return block

        self.model = nn.Sequential(
            *convlayer(self.nz, 1024, 4, 1, 0),  # Fully connected layer via convolution.
            *convlayer(1024, 512, 4, 2, 1),
            *convlayer(512, 256, 4, 2, 1),
            *convlayer(256, 128, 4, 2, 1),
            *convlayer(128, 64, 4, 2, 1),
            *convlayer(64, 32, 4, 2, 1),
            nn.ConvTranspose2d(32, self.channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(-1, self.nz, 1, 1)
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        self.channels = channels

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0, bn=False):
            block = [nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False)]
            if bn:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        self.model = nn.Sequential(
            *convlayer(self.channels, 32, 4, 2, 1),
            *convlayer(32, 64, 4, 2, 1),
            *convlayer(64, 128, 4, 2, 1, bn=True),
            *convlayer(128, 256, 4, 2, 1, bn=True),
            *convlayer(256, 512, 4, 2, 1, bn=True),
            *convlayer(512, 1024, 4, 2, 1, bn=True),
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),  # FC with Conv.
        )

    def forward(self, imgs):
        out = self.model(imgs)
        return out.view(-1, 1)


netG = Generator(nz).to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

fixed_noise = torch.randn(25, nz, 1, 1, device=device)


try:
    netG.load_state_dict(torch.load('./generator/Weights/generator_balo_11_4.h5'))
    netD.load_state_dict(torch.load('./generator/Weights/discriminator_balo_11_4.h5'))
except RuntimeError as e:
    print('Ignoring "' + str(e) + '"')



im_batch_size = 10
n_images=10
for i_batch in range(0, n_images, im_batch_size):
    gen_z = torch.randn(im_batch_size, nz, 1, 1, device=device)
    gen_images = (netG(gen_z) + 1)/2
    images = gen_images.to("cpu").clone().detach()
    images = images.numpy().transpose(0, 2, 3, 1)
    for i_image in range(gen_images.size(0)):
        save_image(gen_images[i_image, :, :, :], os.path.join('./generator/static/generator/images/balo', f'{i_image+1}.png'))


def home(request):
    return render(request, 'generator/home.html')


def images(request):
    path = './generator/static/generator/images/balo/'  # insert the path to your directory
    img_list = []
    img_list = os.listdir(path)
    del img_list[0]
    return render(request,  'generator/images.html', {'title':'Balo', 'category':'balo', 'images': img_list })