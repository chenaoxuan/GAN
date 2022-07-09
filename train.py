"""
Author:  Cax
File:    train
Project: GAN
Time:    2022/7/1
Des:     in this file, the network will be trained
"""
import os
import time
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import make_grid

from GAN.config import cfg
from GAN.data import FashionMNIST
from GAN.data import make_dataloader
from GAN.model import Generator
from GAN.model import Discriminator
from GAN.engine.trainer import train_one_epoch
from GAN.utils.miscellaneous import mkdir_or_exist, save_config
from GAN.utils.logger import get_logger


def final_show(imgs):
    grid = make_grid(imgs, nrow=10, normalize=True).permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid)
    plt.yticks([])
    plt.xticks(np.arange(15, 300, 30),
               ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                'Ankle boot'],
               rotation=45, fontsize=20)
    fig.savefig("./outputs/result.jpg")
    plt.show()


def train(cfg):
    device = cfg.MODEL.DEVICE
    logger = get_logger("GAN.trainer")
    logger.info("Start training")
    dataset = FashionMNIST()
    dataloader = make_dataloader(config=cfg, dataset=dataset)
    generator = Generator(config=cfg).to(device)
    discriminator = Discriminator(config=cfg).to(device)
    criterion = getattr(nn, cfg.SOLVER.LOSS)()
    d_optimizer = getattr(optim, cfg.SOLVER.DISCRIMINATOR.OPTIM) \
        (discriminator.parameters(), lr=cfg.SOLVER.DISCRIMINATOR.LR)
    g_optimizer = getattr(optim, cfg.SOLVER.GENERATOR.OPTIM) \
        (generator.parameters(), lr=cfg.SOLVER.GENERATOR.LR)
    training_time = time.time()
    for epoch in range(1, cfg.SOLVER.MAX_EPOCH + 1):
        train_one_epoch(
            now_epoch=epoch,
            config=cfg,
            dataloader=dataloader,
            generator=generator,
            discriminator=discriminator,
            g_optim=g_optimizer,
            d_optim=d_optimizer,
            criterion=criterion
        )

    generator.eval()
    z = Variable(torch.randn(100, 100)).to(device)
    labels = Variable(torch.LongTensor([i for _ in range(10) for i in range(10)])).to(device)
    sample_imgs = generator(z, labels).unsqueeze(1).data.cpu()
    final_show(sample_imgs)
    training_time = datetime.timedelta(seconds=int(time.time() - training_time))
    logger.info(f"Total training time: {training_time}.")


def main():
    parser = argparse.ArgumentParser(description='Fashion GAN')
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
        type=str,
        required=True
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    mkdir_or_exist(output_dir)
    logger = get_logger(name='GAN', log_dir=output_dir)
    logger.info("A new execution starts")
    logger.info(f"Running with argparse:\n {args}")
    logger.info(f"Running with config:\n {cfg}")
    save_config(config=cfg, save_path=os.path.join(output_dir, 'config.yaml'))

    train(cfg)


if __name__ == '__main__':
    main()
