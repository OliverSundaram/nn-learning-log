from torch import nn

from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(in_features=num_ftrs, out_features=4)