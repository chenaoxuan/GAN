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

import torch.nn as nn
import torch.optim as optim

from GAN.config import cfg
from GAN.data import FashionMNIST
from GAN.data import make_dataloader
from GAN.model import Generator
from GAN.model import Discriminator
from GAN.engine.trainer import train_one_epoch
from GAN.utils.miscellaneous import mkdir_or_exist, save_config
from GAN.utils.logger import get_logger
from GAN.utils.checkpoint import GAN_Checkpointer
from GAN.utils.distributed import is_main_process


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
    checkpointer = GAN_Checkpointer(
        model1=generator,
        model2=discriminator,
        optimizer1=g_optimizer,
        optimizer2=d_optimizer,
        save_to_disk=is_main_process(),
        save_dir=cfg.OUTPUT_DIR,
    )
    if cfg.MODEL.PARAMETER == "":
        old_checkpoint = checkpointer.load(f=None, use_latest=True)
    else:
        old_checkpoint = checkpointer.load(f=cfg.MODEL.PARAMETER, use_latest=False)
    arguments = {'epoch': 1, 'best_epoch': 1, "best_loss": 0}
    arguments.update(old_checkpoint)

    training_time = time.time()
    for epoch in range(arguments['epoch'], cfg.SOLVER.MAX_EPOCH + 1):
        arguments['epoch'] = epoch
        train_one_epoch(
            now_epoch=epoch,
            config=cfg,
            dataloader=dataloader,
            generator=generator,
            discriminator=discriminator,
            g_optim=g_optimizer,
            d_optim=d_optimizer,
            criterion=criterion,
            checkpointer=checkpointer,
            arguments=arguments
        )
        if epoch % cfg.SOLVER.SAVE_EPOCH == 0 \
                and arguments['best_epoch'] != epoch:
            checkpointer.save(name=f'GAN_{epoch}e', **arguments)
    training_time = datetime.timedelta(seconds=int(time.time() - training_time))
    logger.info(f"Total training time: {training_time}.")
    checkpointer.load(
        f=os.path.join(cfg.OUTPUT_DIR, f'GAN_{arguments["best_epoch"]}e_better.pth'),
        use_latest=False
    )


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
    save_config(config=cfg, save_path=os.path.join(cfg.OUTPUT_DIR, 'config.yaml'))

    train(cfg)


if __name__ == '__main__':
    main()
