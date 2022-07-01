"""
Author:  Cax
File:    make_dataloader
Project: GAN
Time:    2022/7/1
Des:     Make dataloader of fashion mnist
"""
import torch


def make_dataloader(config, dataset):
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=config.DATALOADER.BATCH_SIZE,
                                             shuffle=config.DATALOADER.SHUFFLE)
    return dataloader
