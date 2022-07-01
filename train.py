import argparse
import torch.nn as nn
import torch.optim as optim

from GAN.config import cfg
from GAN.data import FashionMNIST
from GAN.data import make_dataloader
from GAN.model import Generator
from GAN.model import Discriminator
from GAN.engine import do_train


def train(cfg):
    device = cfg.MODEL.DEVICE
    dataset = FashionMNIST()
    dataloader = make_dataloader(config=cfg, dataset=dataset)
    generator = Generator(config=cfg).to(device)
    discriminator = Discriminator(config=cfg).to(device)
    criterion = getattr(nn, cfg.SOLVER.LOSS)()
    d_optimizer = getattr(optim, cfg.SOLVER.DISCRIMINATOR.OPTIM) \
        (discriminator.parameters(), lr=cfg.SOLVER.DISCRIMINATOR.LR)
    g_optimizer = getattr(optim, cfg.SOLVER.GENERATOR.OPTIM) \
        (generator.parameters(), lr=cfg.SOLVER.GENERATOR.LR)

    best_epoch = do_train(config=cfg,
                          dataloader=dataloader,
                          generator=generator,
                          g_optim=g_optimizer,
                          discriminator=discriminator,
                          d_optim=d_optimizer,
                          criterion=criterion)


def main():
    parser = argparse.ArgumentParser(description='Fashion GAN')
    parser.add_argument(
        "--config-file",
        default="configs/baseline.yaml",
        metavar="FILE",
        help="path to config file",
        type=str
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    model = train(cfg)


if __name__ == '__main__':
    main()
