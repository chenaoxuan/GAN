"""
Author:  Cax
File:    test
Project: GAN
Time:    2022/7/3
Des:     
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import make_grid

from GAN.config import cfg
from GAN.model import Generator
from GAN.model import Discriminator
from GAN.utils.logger import get_logger
from GAN.utils.checkpoint import GAN_Checkpointer
from GAN.utils.distributed import is_main_process
from GAN.utils.miscellaneous import mkdir_or_exist


def test(cfg):
    epoch = 50
    logger = get_logger('GAN.test')
    logger.info("Start test")
    generator = Generator(config=cfg)
    discriminator = Discriminator(config=cfg)
    d_optimizer = getattr(optim, cfg.SOLVER.DISCRIMINATOR.OPTIM) \
        (discriminator.parameters(), lr=cfg.SOLVER.DISCRIMINATOR.LR)
    g_optimizer = getattr(optim, cfg.SOLVER.GENERATOR.OPTIM) \
        (generator.parameters(), lr=cfg.SOLVER.GENERATOR.LR)
    checkpointer = GAN_Checkpointer(
        model1=generator,
        model2=discriminator,
        optimizer1=g_optimizer,
        optimizer2=d_optimizer,
        save_to_disk=is_main_process(),
        save_dir=cfg.OUTPUT_DIR
    )
    dic = checkpointer.load(
        f=os.path.join(cfg.OUTPUT_DIR, f'GAN_{epoch}e_better.pth'),
        use_latest=False
    )
    print(dic)
    generator.eval()
    z = Variable(torch.randn(100, 100))
    labels = Variable(torch.LongTensor([i for _ in range(10) for i in range(10)]))
    imgs = generator(z, labels).unsqueeze(1).data
    grid = make_grid(imgs, nrow=10, normalize=True).permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid)
    plt.yticks([])
    plt.xticks(np.arange(15, 300, 30),
               ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                'Ankle boot'],
               rotation=45, fontsize=20)
    fig.savefig(f"./outputs/result{epoch}.jpg")
    plt.show()


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

    mkdir_or_exist(cfg.OUTPUT_DIR)
    logger = get_logger(name='GAN', log_dir=cfg.OUTPUT_DIR)
    logger.info("A new execution starts")
    logger.info(f"Running with argparse:\n {args}")
    logger.info(f"Running with config:\n {cfg}")

    test(cfg)


if __name__ == '__main__':
    main()
