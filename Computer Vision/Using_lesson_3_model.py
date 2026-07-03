import torch
from torch import nn

from matplotlib import pyplot as plt

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from helper_functions import accuracy_fn
from timeit import default_timer as timer

KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1

class FashionMNISTModel(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=KERNEL_SIZE,
                      stride=STRIDE,
                      padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=KERNEL_SIZE,
                      stride=STRIDE,
                      padding=PADDING),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=KERNEL_SIZE,
                      stride=STRIDE,
                      padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=KERNEL_SIZE,
                      stride=STRIDE,
                      padding=PADDING),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=output_shape*49,
                      out_features=10)
        )

    def forward(self, x: torch.Tensor):
        z = self.conv_block_1(x)
        z = self.conv_block_2(z)
        z = self.classifier(z)
        return z

test_data = datasets.FashionMNIST(root="data",
                                  train=False,
                                  download=True,
                                  transform=ToTensor(),
                                  target_transform=None)
test_data = DataLoader(dataset=test_data,
                       batch_size=1,
                       shuffle=True)

class_names = datasets.FashionMNIST.classes

model = FashionMNISTModel(1, 10, len(class_names))
model.load_state_dict(torch.load(f="models/CNN_for_FashionMNIST.pth"))

while True:
    data = next(iter(test_data))
    X, y = data

    y_logits = model(X)
    y_probs = torch.softmax(y_logits, dim=1)
    y_pred = torch.argmax(y_probs, dim=1)

    plt.imshow(X.squeeze(), cmap="gray")
    plt.axis(False)
    print(f"Label: {class_names[y.item()]} | AI guess: {class_names[y_pred.item()]}")
    plt.show()