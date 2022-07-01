"""
Author:  Cax
File:    base
Project: GAN
Time:    2022/7/1
Des:     Store basic variables of generator and discriminator
"""

import torch.nn as nn


class Base(nn.Module):
    def __init__(self, config):
        super(Base, self).__init__()
        self.lnum = config.DATASETS.LAB.NUM
        self.fdim = config.MODEL.GENERATOR.FAKE_DIM
        self.idim = config.DATASETS.IMG.DIM
