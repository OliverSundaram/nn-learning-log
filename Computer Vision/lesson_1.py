# Computer vision libraries

# Torch vision - Popular library that has many datasets for computer vision
# torchvision.datasets - get datasets and data loading functions for computer vision
# torchvision.models - get pretrained computer vision models
# torchvision.transforms - functions for manipulating vision data(images) to be suitable
# torch.utils.data.Dataset - Base dataset class for PyTorch
# torch.utils.data.DataLoader - Creates a Python iterable over a dataset



import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import models
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt



# Getting a dataset
# fashionMNIST - dataset containing grey-scale images of clothes
train_data = datasets.FashionMNIST(root="data", # where to download data to
                                   train=True, # do we want training or testing
                                   download=True,
                                   transform=ToTensor(),
                                   target_transform=None)
test_data = datasets.FashionMNIST(root="data",
                                  train=False,
                                  download=True,
                                  transform=ToTensor(),
                                  target_transform=None)

# image, label = train_data[0]
# plt.imshow(image.squeeze(), cmap="gray")
# plt.show()
#
# torch.manual_seed(42)
# print(image.shape, label)

# Prepare DataLoader
# DataLoader - turns dataset into a python iterable
# Want to turn data into batches
# Why?
# 1. Efficient, breaking down data into batch sizes(e.g., 32)
# 2. Gives model more times to update parameters

# Setup batch size hyperparameter(parameter you set)
BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

train_features_batch, train_labels_batch = next(iter(train_dataloader))

# print(len(train_dataloader), len(test_dataloader))
# torch.manual_seed(42)
random_inx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_inx], train_labels_batch[random_inx]
plt.imshow(img.squeeze(), cmap="gray")
plt.axis(False)
print(label)
plt.show()