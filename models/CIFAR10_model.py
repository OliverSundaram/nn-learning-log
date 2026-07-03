import torch
from torch import nn

from matplotlib import pyplot as plt

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from helperfunctions.helper_functions import accuracy_fn

train_data = datasets.CIFAR10(root="data",
                              train=True,
                              transform=ToTensor(),
                              target_transform=None,
                              download=True)
test_data = datasets.CIFAR10(root="data",
                             train=False,
                             transform=ToTensor(),
                             target_transform=None,
                             download=True)

class_names = train_data.classes

train_dataloader = DataLoader(dataset=train_data,
                        batch_size=16,
                        shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                       batch_size=16,
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
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.01)

epochs = 5
for epoch in range(1, epochs + 1):

    train_loss, train_acc = 0, 0
    model.train()
    for data_batch in train_dataloader:
        X, y = data_batch
        y_logits = model(X)
        y_preds = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
        loss = loss_fn(y_logits, y)
        acc = accuracy_fn(y, y_preds)
        train_loss += loss
        train_acc += acc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    model.eval()
    with torch.inference_mode():
        test_loss, test_acc = 0, 0
        for data_batch in test_dataloader:
            X, y = data_batch
            y_logits = model(X)
            y_preds = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
            loss = loss_fn(y_logits, y)
            acc = accuracy_fn(y, y_preds)
            test_loss += loss
            test_acc += acc
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    print(f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}% | Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}%")



# Saving PyTorch model
from pathlib import Path

# 1. Create model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "CNN_for_CIFAR10.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)