import torch
from torch import nn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from notes.helper_functions.Plot_decision_boundary import plot_decision_boundary as pdb
from torchmetrics import Accuracy

device = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 7
accuracy = Accuracy(task="multiclass", num_classes=6).to(device)

X, y = make_blobs(n_samples=1000,
                  n_features=2,
                  centers=6,
                  cluster_std=0.6,
                  random_state=RANDOM_STATE)

X = torch.Tensor(X).to(device)
y = torch.Tensor(y).to(device).type(torch.LongTensor)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=RANDOM_STATE)
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

class BlobModel(nn.Module):
    def __init__(self, input, hidden, hidden_2, output):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)

model = BlobModel(input=2, hidden=32, hidden_2=64, output=6).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.001)

epochs = 10000
for epoch in range(1, epochs + 1):
    model.train()
    y_logits = model(X_train)
    y_preds = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
    train_acc = accuracy(y_preds, y_train)
    train_loss = loss_fn(y_logits, y_train)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    if (epoch % (epochs / 10)) == 0 or epoch == 1:
        model.eval()
        with torch.inference_mode():
            y_test_logits = model(X_test)
            y_preds = torch.argmax(torch.softmax(y_test_logits, dim=1), dim=1)
            test_acc = accuracy(y_preds, y_test)
            test_loss = loss_fn(y_test_logits, y_test)
        print(f"Epoch: {epoch} | Train loss: {round(train_loss.item(), 4)}, Train Acc: {round(train_acc.item() * 100, 4)}% | Test loss: {round(test_loss.item(), 4)}, Test Acc: {round(test_acc.item() * 100, 4)}%")

time.sleep(1)
print("Making predictions...")
time.sleep(1)

model.eval()
with torch.inference_mode():
    y_logits = model(X_test)
    y_prob = torch.softmax(y_logits, dim=1)
    y_preds = torch.argmax(y_prob, dim=1)

print(f"First 6 answers to data: {y_test[:6]}, First 6 predictions of model: {y_preds[:6]}")

time.sleep(1)
print("Showing Plot Decision Boundary...")
time.sleep(1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Train Decision Boundary")
pdb(model, X_train, y_train, device)

plt.subplot(1, 2, 2)
plt.title("Test Decision Boundary")
pdb(model, X_test, y_test, device)
plt.show()