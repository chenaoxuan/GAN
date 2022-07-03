import os
import argparse
import torch.nn as nn
import torch.optim as optim

from GAN.config import cfg
from GAN.data import FashionMNIST
from GAN.data import make_dataloader
from GAN.model import Generator
from GAN.model import Discriminator
from GAN.engine import do_train

from GAN.utils.miscellaneous import mkdir_or_exist, save_config
from GAN.utils.logger import get_logger


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

    model = train(cfg)


if __name__ == '__main__':
    main()
