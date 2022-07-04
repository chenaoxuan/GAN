"""
Author:  Cax
File:    trainer
Project: GAN
Time:    2022/7/1
Des:     in this file, the network will be trained
"""
import datetime
import time

import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from GAN.utils.logger import get_logger
from GAN.utils.metric_logger import MetricLogger


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


def do_train(config, dataloader, generator, g_optim, discriminator, d_optim, criterion):
    device = config.MODEL.DEVICE
    max_epoch = config.SOLVER.MAX_EPOCH
    logger = get_logger("GAN.trainer")
    logger.info("Start training")
    mlogger = MetricLogger()
    max_iteration = len(dataloader)
    print_every_iter = max_iteration // 2
    training_time = end = time.time()
    for epoch in range(1, max_epoch + 1):
        for iteration, (images, labels) in enumerate(dataloader, 1):
            real_imgs = Variable(images).to(device)
            labels = Variable(labels).to(device)
            generator.train()
            d_loss = discriminator_train_step(config,
                                              discriminator,
                                              generator,
                                              d_optim,
                                              criterion,
                                              real_imgs,
                                              labels)
            g_loss = generator_train_step(config,
                                          discriminator,
                                          generator,
                                          g_optim,
                                          criterion)
            mlogger.update(d_loss=d_loss, g_loss=g_loss)
            batch_time = time.time() - end
            end = time.time()
            mlogger.update(batch_time=batch_time)
            eta_seconds = mlogger.batch_time.global_avg * (
                    max_iteration - iteration + (max_epoch - epoch) * max_iteration)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            if iteration % print_every_iter == 0 or iteration == max_iteration:
                log_msg = [
                    f"eta: {eta}",
                    f"epoch: {epoch}/{max_epoch}",
                    f"iteration: {iteration}/{max_iteration}",
                    f"{str(mlogger)}"
                ]
                if real_imgs.is_cuda:
                    log_msg.append(f"memory: {torch.cuda.max_memory_allocated() / 1024.0 / 1024.0:.0f}MB")
                logger.info(mlogger.delimiter.join(log_msg))
        generator.eval()
        # step_show(sample_imgs)
    training_time = time.time() - training_time
    training_time = str(datetime.timedelta(seconds=int(training_time)))
    logger.info(f"Total training time: {training_time}.")

    # final_show(sample_imgs)


def step_show(imgs):
    grid = make_grid(imgs, nrow=3, normalize=True).permute(1, 2, 0).numpy()
    plt.imshow(grid)
    plt.show()


def final_show(imgs):
    grid = make_grid(imgs, nrow=10, normalize=True).permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid)
    _ = plt.yticks([])
    _ = plt.xticks(np.arange(15, 300, 30),
                   ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                    'Ankle boot'],
                   rotation=45, fontsize=20)
    plt.show()
