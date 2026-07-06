from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch import nn


def accuracy_fn(y_true, y_preds):
    correct = torch.eq(y_true, y_preds).sum().item()
    acc = (correct/len(y_preds)) * 100
    return acc


num_samples = 1000

# Create circles
X, y = make_circles(num_samples, # Number of CIFAR10_data
                    noise=0.03, # Add noise to the CIFAR10_data points
                    random_state=42) # Set random seed

# Change CIFAR10_data into tensors, as they were returned as numpy arrays
X = torch.Tensor(X)
y = torch.Tensor(y)

# Split CIFAR10_data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Setup device agnostic code so model can run on GPU if there is one
device = "cuda" if torch.cuda.is_available() else "cpu"

X_train = torch.Tensor(X_train).to(device)
X_test = torch.Tensor(X_test).to(device)
y_train = torch.Tensor(y_train).to(device)
y_test = torch.Tensor(y_test).to(device)

# Create a model with 2 inputs, 5 neurons in hidden layer, 1 output

model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=8),
    nn.Linear(in_features=8, out_features=1)
).to(device)

# Setup loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss() # Contains the sigmoid activation function built in
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

torch.cuda.manual_seed(42)

epochs = 100

for epoch in range(1, epochs + 1):
    model_0.train()

    # Forward pass
    y_logits = model_0(X_train).squeeze()
    y_preds = torch.round(torch.sigmoid(y_logits)) # Turn logits into probabilities into labels

    # Calculate loss/accuracy
    loss = loss_fn(y_logits, # nn.BCEWIthLogitsLoss expects raw logits as input. DOes sigmoid by its self
                   y_train)

    acc = accuracy_fn(y_true=y_train, y_preds=y_preds)

    optimizer.zero_grad()

    # Backpropagation - Calculate gradients
    loss.backward()

    # Gradient descent - Update parameters to reduce loss
    optimizer.step()

    # Testing
    model_0.eval()
    with torch.inference_mode():

        # Forward pass
        test_logits = model_0(X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))

        # Calculate loss/acc
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_preds=test_preds)

        # Print what's happening
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss:{loss} | Acc: {acc}% | Test loss: {test_loss} | Test acc: {test_acc}%")