import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import models
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

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

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

train_features_batch, train_labels_batch = next(iter(train_dataloader))

# # Create a flatten layer
# flatten_model = nn.Flatten()
#
# # Get a single sample
# x = train_features_batch[0]
# print(x.shape)
#
# # Flatten the sample
# output = flatten_model(x)
#
# print(output.shape)

class FashionMNISTModel(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)

torch.manual_seed(42)
model = FashionMNISTModel(
    input_shape=784,
    hidden_units=16,
    output_shape=len(class_names)
).to("cpu")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.02)

# import requests
# from pathlib import Path
#
# if Path("helper_functions.py").is_file():
#     print("Already exists")
# else:
#     print("Downloading helper_functions.py")
#     request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
#     with open("helper_functions.py", "wb") as f:
#         f.write(request.content)

from helper_functions import accuracy_fn

from timeit import default_timer as timer
def print_train_time(start: float,
                     end: float,
                     device: str = None):
    """Prints difference between start and end time."""
    total_time = end - start
    print(f"Train time on {device}: {total_time} seconds")
    return total_time




start_time = timer()

from tqdm.auto import tqdm
torch.manual_seed(42)

epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"\nEpoch: {epoch}\n")

    train_loss = 0
    # Add a loop to loop through the training batches
    for batch, (X_train, y_train) in enumerate(train_dataloader):
        model.train()
        y_preds = model(X_train)
        loss = loss_fn(y_preds, y_train)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 400 == 0:
            print(f"Looked at: {batch * len(X_train)}/{len(train_dataloader.dataset)} samples")

    train_loss /= len(train_dataloader)

    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            test_pred = model(X_test)
            test_loss += loss_fn(test_pred, y_test)

            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    print(f"Train loss: {train_loss} | Test loss: {test_loss}, Test acc: {test_acc}%")

end_time = timer()
total_train_time = print_train_time(start=start_time, end=end_time, device="cpu")