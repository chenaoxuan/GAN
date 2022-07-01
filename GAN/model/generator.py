"""
Author:  Cax
File:    generator
Project: GAN
Time:    2022/6/30
Des:     Generator in GAN
"""
import torch
import torch.nn as nn
from .base import Base


class Generator(Base):
    def __init__(self, config):
        super(Generator, self).__init__(config)
        self.lemb = config.MODEL.GENERATOR.LABEL_EMB
        self.label_emb = nn.Embedding(self.lnum, self.lemb)
        self.model = nn.Sequential(
            nn.Linear(self.fdim + self.lemb, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.idim ** 2),
            nn.Tanh()
        )

    def forward(self, x, labels):
        """
        Generator
        :param x: fake seeds with dimension[bs,fdim]
        :param labels: labels with dimension[bs]
        :return:[bs,idim,idim]
        """
        l = self.label_emb(labels)  # [batch_size,lemb]
        x = torch.cat([x, l], 1)  # [batch_size,lemb+fdim]
        x = self.model(x)
        return x.view(x.size(0), self.idim, self.idim)
