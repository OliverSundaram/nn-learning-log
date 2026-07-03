import torch
from torch import nn

from matplotlib import pyplot as plt

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from timeit import default_timer as timer
from helperfunctions.helper_functions import accuracy_fn

test_data = datasets.CIFAR10(root="data",
                             train=False,
                             transform=ToTensor(),
                             target_transform=None,
                             download=True)

class_names = test_data.classes

test_dataloader = DataLoader(dataset=test_data,
                       batch_size=1,
                       shuffle=True)

KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1

class CIFAR10Model(nn.Module):
    def __init__(self, input_shape, hidden_shape, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_shape,
                      kernel_size=KERNEL_SIZE,
                      stride=STRIDE,
                      padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_shape,
                      out_channels=hidden_shape,
                      kernel_size=KERNEL_SIZE,
                      stride=STRIDE,
                      padding=PADDING),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_shape,
                      out_channels=hidden_shape,
                      kernel_size=KERNEL_SIZE,
                      stride=STRIDE,
                      padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_shape,
                      out_channels=hidden_shape,
                      kernel_size=KERNEL_SIZE,
                      stride=STRIDE,
                      padding=PADDING),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_shape*8*8,
                      out_features=output_shape)
        )

    def forward(self, x):
        z = self.conv_block_1(x)
        z = self.conv_block_2(z)
        z = self.classifier(z)
        return z


model = CIFAR10Model(3, 16, len(class_names))
model.load_state_dict(torch.load(f="models/CNN_for_CIFAR10.pth"))

while True:
    data = next(iter(test_dataloader))
    X, y = data
    y_logits = model(X)
    y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)

    X = X.squeeze()
    X = torch.permute(X, dims=[1, 2, 0])

    plt.imshow(X.squeeze())
    plt.axis(False)
    print(f"Label: {class_names[y.item()]} | AI guess: {class_names[y_pred.item()]}")
    plt.show()