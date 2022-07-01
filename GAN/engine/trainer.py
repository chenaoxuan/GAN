"""
Author:  Cax
File:    trainer
Project: GAN
Time:    2022/7/1
Des:     in this file, the network will be trained
"""
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def generator_train_step(config, batch_size, discriminator, generator, g_optimizer, criterion):
    device = config.MODEL.DEVICE
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 100)).to(device)
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).to(device)
    fake_images = generator(z, fake_labels)
    validity = discriminator(fake_images, fake_labels)
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).to(device))
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()


def discriminator_train_step(config, batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
    device = config.MODEL.DEVICE
    d_optimizer.zero_grad()

    # train with real images
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(device))

    # train with fake images
    z = Variable(torch.randn(batch_size, 100)).to(device)
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).to(device)
    fake_images = generator(z, fake_labels)
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).to(device))

    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()


def do_train(config, dataloader, generator, g_optim, discriminator, d_optim, criterion):
    device = config.MODEL.DEVICE
    for epoch in range(config.SOLVER.MAX_EPOCH):
        print(f'Starting epoch {epoch}')
        for i, (images, labels) in enumerate(dataloader):
            real_images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            generator.train()
            batch_size = real_images.size(0)
            d_loss = discriminator_train_step(config, len(real_images), discriminator,
                                              generator, d_optim, criterion,
                                              real_images, labels)

            g_loss = generator_train_step(config, batch_size, discriminator, generator, g_optim, criterion)

        generator.eval()
        print('g_loss: {}, d_loss: {}'.format(g_loss, d_loss))
        z = Variable(torch.randn(9, 100)).to(device)
        labels = Variable(torch.LongTensor(np.arange(9))).to(device)
        sample_images = generator(z, labels).unsqueeze(1).data.cpu()
        grid = make_grid(sample_images, nrow=3, normalize=True).permute(1, 2, 0).numpy()
        plt.imshow(grid)
        plt.show()
    z = Variable(torch.randn(100, 100)).to(device)
    labels = Variable(torch.LongTensor([i for _ in range(10) for i in range(10)])).to(device)
    sample_images = generator(z, labels).unsqueeze(1).data.cpu()
    grid = make_grid(sample_images, nrow=10, normalize=True).permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(grid)
    _ = plt.yticks([])
    _ = plt.xticks(np.arange(15, 300, 30),
                   ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                    'Ankle boot'],
                   rotation=45, fontsize=20)
    plt.show()
