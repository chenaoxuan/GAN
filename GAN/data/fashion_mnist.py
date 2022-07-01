"""
Author:  Cax
File:    fashion_mnist
Project: GAN
Time:    2022/7/1
Des:     Create Fashion MNIST dataset
"""
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("GAN\\") + len("GAN\\")]
dataPath = os.path.abspath(rootPath + 'datasets')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])


class FashionMNIST(Dataset):
    def __init__(self):
        self.transform = transform
        fashion_df = pd.read_csv(os.path.join(dataPath, 'fashion-mnist_train.csv'))
        self.labels = fashion_df.label.values
        self.images = fashion_df.iloc[:, 1:].values.astype('uint8').reshape(-1, 28, 28)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img, label


if __name__ == '__main__':
    print(curPath)
    print(dataPath)
    # dataset = FashionMNIST()
