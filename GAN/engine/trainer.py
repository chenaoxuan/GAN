"""
Author:  Cax
File:    trainer
Project: GAN
Time:    2022/7/1
Des:     Training will iterate only one epoch.
"""
import time
import datetime
import numpy as np

import torch
from torch.autograd import Variable

from GAN.utils.logger import get_logger
from GAN.utils.metric_logger import MetricLogger, SmoothedValue


def generator_train_step(config, discriminator, generator, g_optimizer, criterion):
    """
    generate fake images and train generator by cheating discriminator.
    """
    device = config.MODEL.DEVICE
    batch_size = config.DATALOADER.BATCH_SIZE
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 100)).to(device)
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).to(device)
    fake_images = generator(z, fake_labels)
    validity = discriminator(fake_images, fake_labels)
    # cheat discriminator
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).to(device))
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()


def discriminator_train_step(config, discriminator, generator, d_optimizer, criterion, real_images, labels):
    """
    use real images and generated fake images to train discriminator
    """
    device = config.MODEL.DEVICE
    batch_size = len(real_images)
    d_optimizer.zero_grad()

    # train with real images
    real_val = discriminator(real_images, labels)
    real_loss = criterion(real_val, Variable(torch.ones(batch_size)).to(device))

    # train with fake images
    z = Variable(torch.randn(batch_size, 100)).to(device)
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).to(device)
    fake_imgs = generator(z, fake_labels)
    fake_val = discriminator(fake_imgs, fake_labels)
    fake_loss = criterion(fake_val, Variable(torch.zeros(batch_size)).to(device))

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()


def train_one_epoch(now_epoch, config, dataloader, generator, g_optim, discriminator, d_optim, criterion):
    mlogger = MetricLogger()
    device = config.MODEL.DEVICE
    logger = get_logger("GAN.trainer")
    max_iteration = len(dataloader)
    print_every_iter = max_iteration // 2
    end = time.time()
    mlogger.add_meter("batch_time", SmoothedValue(avg_only=True))
    for iteration, (imgs, labels) in enumerate(dataloader, 1):
        real_imgs = Variable(imgs).to(device)
        labels = Variable(labels).to(device)
        generator.train()
        d_loss = discriminator_train_step(
            config=config,
            discriminator=discriminator,
            generator=generator,
            d_optimizer=d_optim,
            criterion=criterion,
            real_images=real_imgs,
            labels=labels
        )
        g_loss = generator_train_step(
            config=config,
            generator=generator,
            discriminator=discriminator,
            g_optimizer=g_optim,
            criterion=criterion,
        )
        mlogger.update(d_loss=d_loss, g_loss=g_loss, batch_time=time.time() - end)
        end = time.time()
        ets_seconds = mlogger.batch_time.global_avg * (
                max_iteration - iteration
                + (config.SOLVER.MAX_EPOCH - now_epoch) * max_iteration
        )
        eta = datetime.timedelta(seconds=int(ets_seconds))
        if iteration % print_every_iter == 0 or iteration == max_iteration:
            log_msg = [
                f"eta: {eta}",
                f"epoch: {now_epoch}/{config.SOLVER.MAX_EPOCH}",
                f"iteration: {iteration}/{max_iteration}",
                f"{str(mlogger)}"
            ]
            if real_imgs.is_cuda:
                log_msg.append(f"memory: {torch.cuda.max_memory_allocated() / 1024.0 / 1024.0:.0f}MB")
            logger.info(mlogger.delimiter.join(log_msg))
    generator.eval()
