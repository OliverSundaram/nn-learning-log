import torch
from torch import nn

from matplotlib import pyplot as plt

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from helper_functions import accuracy_fn
from timeit import default_timer as timer

from helper_functions import accuracy_fn

def print_train_time(start, end, device):
    """Prints & Returns difference between start and end time (end - start)."""
    total_time = end - start
    print(f"Train time on {device}: {total_time} seconds")
    return total_time



# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

train_data = datasets.FashionMNIST(root="CIFAR10_data",
                                   train=True,
                                   download=True,
                                   transform=ToTensor(),
                                   target_transform=None)
test_data = datasets.FashionMNIST(root="CIFAR10_data",
                                  train=False,
                                  download=True,
                                  transform=ToTensor(),
                                  target_transform=None)

class_names = datasets.FashionMNIST.classes

BATCH_SIZE = 32

train_data = DataLoader(dataset=train_data,
                        batch_size=BATCH_SIZE,
                        shuffle=True)
test_data = DataLoader(dataset=test_data,
                       batch_size=BATCH_SIZE,
                       shuffle=True)

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


torch.manual_seed(0)
model = FashionMNISTModel(1, 10, len(class_names))
model.load_state_dict(torch.load(f="models_state_dict/CNN_for_FashionMNIST.pth"))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.03)

start_time = timer()
epochs = 50

for epoch in range(1, epochs + 1):

    train_loss, train_acc = 0, 0
    model.train()
    for batch, (X, y) in enumerate(train_data):
        y_logits = model(X)
        y_preds = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
        loss = loss_fn(y_logits, y)
        acc = accuracy_fn(y, y_preds)
        train_loss += loss
        train_acc += acc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_data)
    train_acc /= len(train_data)

    if (epoch % (epochs / 50)) == 0 or epoch == 1:
        model.eval()
        with torch.inference_mode():
            test_loss, test_acc = 0, 0
            for (X, y) in test_data:
                y_logits = model(X)
                y_preds = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
                loss = loss_fn(y_logits, y)
                acc = accuracy_fn(y, y_preds)
                test_loss += loss
                test_acc += acc
            test_loss /= len(test_data)
            test_acc /= len(test_data)
            print(f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}%, Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}%")

end_time = timer()
print_train_time(start_time, end_time, "CPU")



# Saving PyTorch model
from pathlib import Path

# 1. Create model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "CNN_for_FashionMNIST.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)