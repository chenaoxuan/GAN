"""
Author:  Cax
File:    discriminator
Project: GAN
Time:    2022/7/1
Des:     Discriminator in GAN
"""
import torch
import torch.nn as nn
from .base import Base


class Discriminator(Base):
    def __init__(self, config):
        super(Discriminator, self).__init__(config)
        self.lemb = config.MODEL.DISCRIMINATOR.LABEL_EMB
        self.label_emb = nn.Embedding(self.lnum, self.lemb)
        self.model = nn.Sequential(
            nn.Linear(self.idim ** 2 + self.lemb, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, imgs, labels):
        """
        Discriminator
        :param imgs:image with dimension[bs,idim,idim]
        :param labels:labels with dimension[bs]
        :return:[bs]
        """
        i = imgs.view(imgs.size(0), self.idim ** 2)  # [bs,idim**2]
        l = self.label_emb(labels)  # [bs,lemb]
        x = torch.cat([i, l], 1)  # [bs,idim**2+lemb]
        x = self.model(x)  # [bs,1]
        return x.squeeze()
