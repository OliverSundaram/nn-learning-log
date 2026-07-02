import torch
from torch import nn

from matplotlib import pyplot as plt

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from helper_functions import accuracy_fn
from timeit import default_timer as timer





# Making training and testing functions
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: str):
    """Performs training on a given model to learn from data_loader."""
    train_loss, train_acc = 0, 0
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_logits = model(X)
        y_preds = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
        loss = loss_fn(y_logits, y)
        train_loss += loss
        train_acc += accuracy_fn(y, y_preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    print(f"Train loss: {train_loss} | Train acc: {train_acc}%")

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: str):
    """Performs a testing loop step on model going over data_loader"""
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            y_preds = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
            test_loss += loss_fn(y_logits, y)
            test_acc += accuracy_fn(y, y_preds)

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

    print(f"Test loss: {test_loss} | Test Accuracy: {test_acc}%")





# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

def print_train_time(start, end, device):
    """Prints & Returns difference between start and end time (end - start)."""
    total_time = end - start
    print(f"Train time on {device}: {total_time} seconds")
    return total_time

train_data = datasets.FashionMNIST(root="data",
                                   train=True,
                                   download=True,
                                   transform=ToTensor(),
                                   target_transform=None)
test_data = datasets.FashionMNIST(root="data",
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

class FashionMNISTModel(nn.Module):
    def __init__(self,
                 input_shape,
                 hidden_shape_1,
                 hidden_shape_2,
                 output_shape):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_shape_1),
            nn.ReLU(),
            nn.Linear(in_features=hidden_shape_1, out_features=hidden_shape_2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_shape_2, out_features=hidden_shape_1),
            nn.ReLU(),
            nn.Linear(in_features=hidden_shape_1, out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer_stack(x)

torch.manual_seed(0)
model = FashionMNISTModel(784,32,64, len(class_names)).to(device)
# # Load the saved state dict of the saved model (this will update the model with updated parameters)
# model.load_state_dict(torch.load(f="models/FashionMNISTModel.pth"))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.1)

# Train loop
start_time = timer()
torch.manual_seed(0)
torch.cuda.manual_seed(0)

epochs = 10
for epoch in range(1, epochs + 1):
    print(f"Epoch: {epoch}\n---------")

    train_step(model=model, data_loader=train_data, loss_fn=loss_fn, optimizer=optimizer, accuracy_fn=accuracy_fn, device=device)

    test_step(model=model, data_loader=test_data, loss_fn=loss_fn, accuracy_fn=accuracy_fn, device=device)

end_time = timer()
print_train_time(start_time, end_time, device=device)
#
#
# # Saving PyTorch model
# from pathlib import Path
#
# # 1. Create model directory
# MODEL_PATH = Path("models")
# MODEL_PATH.mkdir(parents=True, exist_ok=True)
#
# # 2. Create model save path
# MODEL_NAME = "FashionMNISTModel.pth"
# MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
#
# # 3. Save the model state dict
# print(f"Saving model to: {MODEL_SAVE_PATH}")
# torch.save(obj=model.state_dict(),
#            f=MODEL_SAVE_PATH)