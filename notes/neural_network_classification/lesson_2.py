from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch import nn
import random
from helper_functions.Plot_decision_boundary import plot_decision_boundary as pdb
device = "cuda" if torch.cuda.is_available() else "cpu"

def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_true)) * 100
    return acc

# Creating a toy multi-class dataset

# Set the hyperparameters for CIFAR10_data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# Create multi-class CIFAR10_data
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1, # Gives clusters a shake-up
                            random_state=RANDOM_SEED)

X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

# Split into train and test
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

def plot_predictions(X_blob_train=X_blob_train, X_blob_test=X_blob_test, y_blob_train=y_blob_train, y_blob_test=y_blob_test, predictions=None):
    plt.figure()
    plt.scatter(X_blob_train[:, 0], X_blob_train[:, 1], c=y_blob_train)
    plt.show()



class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)

model = BlobModel(input_features=2, output_features=4, hidden_units=8).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.01)

X_blob_train = X_blob_train.to(device)
X_blob_test = X_blob_test.to(device)
y_blob_train = y_blob_train.to(device)
y_blob_test = y_blob_test.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Training loop
epochs = 1000
for epoch in range(1, epochs + 1):

    model.train()
    y_logits = model(X_blob_train)
    y_pred_probs = torch.softmax(y_logits, dim=1)
    y_preds = torch.argmax(y_pred_probs, dim=1)

    train_acc = accuracy(y_blob_train, y_preds)
    train_loss = loss_fn(y_logits, y_blob_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Testing
    if (epoch % (epochs / 10)) == 0 or epoch == 1:
        model.eval()
        with torch.inference_mode():
            y_test_logits = model(X_blob_test)
            y_pred_probs = torch.softmax(y_test_logits, dim=1)
            y_preds = torch.argmax(y_pred_probs, dim=1)

            test_acc = accuracy(y_blob_test, y_preds)
            test_loss = loss_fn(y_test_logits, y_blob_test)

        print(f"Epoch: {epoch} | Train loss: {train_loss} | Train Acc: {train_acc}% | Test loss: {test_loss} | Test Acc: {test_acc}%")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Train Decision Boundary")
pdb(model, X_blob_train, y_blob_train, device)

plt.subplot(1, 2, 2)
plt.title("Test Decision Boundary")
pdb(model, X_blob_test, y_blob_test, device)
plt.show()