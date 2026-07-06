from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch import nn
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

data_possibilities = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

def accuracy_fn(y_true: torch.Tensor, y_preds: torch.Tensor) -> float:
    correct = torch.eq(y_true, y_preds).sum().item()
    return correct * 100/len(y_true)

def gen_data(size: int = 100) -> torch.Tensor:
    data = []
    for i in range(size):
        data.append(random.choice(data_possibilities))
    return torch.tensor(data, dtype=torch.float).to(device)

weights = torch.tensor([1, 2, 3], dtype=torch.float).to(device)
bias = torch.tensor(0.0, dtype=torch.float).to(device)

X = gen_data()
y = torch.matmul(X, weights) + bias

# Split CIFAR10_data
t_s = int(0.8 * len(X))

X_train = X[:t_s].to(device)
X_test = X[t_s:].to(device)
y_train = y[:t_s].to(device)
y_test = y[t_s:].to(device)

class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=3, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))

model = CircleModel().to(device=device)
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.001)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 10000

for epoch in range(1, epochs + 1):
    # Training
    model.train()
    y_preds = model(X_train).squeeze()
    train_loss = loss_fn(y_train, y_preds)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Testing
    if (epoch % (epochs / 10)) == 0 or epoch == 1:
        model.eval()

        with torch.inference_mode():
            test_preds = model(X_test).squeeze()
            test_loss = loss_fn(y_test, test_preds)

            print(f"Epoch: {epoch} | Train loss: {train_loss} | Test loss: {test_loss}")