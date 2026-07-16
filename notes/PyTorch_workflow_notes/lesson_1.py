import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

# 1. Data (preparing and loading)

# Data can be almost anything in ML
# Spead sheet
# Images
# Videos
# Audio
# DNA
# Text

# ML has two parts
# 1. Get CIFAR10_data into a numerical representation
# 2. Build a model to learn patterns in that CIFAR10_data



# Create known parameters
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(1)
y = weight * X + bias

# print(X[:10], y[:10])
# print(len(X), len(y))


# Splitting CIFAR10_data into training and test sets


# Create a train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# print(len(X_train), len(y_train), len(X_test), len(y_test))


# Visualizing the CIFAR10_data

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None) -> None:
    """
    Plots training CIFAR10_data, test CIFAR10_data and compares predictions
    """

    plt.figure()

    # Plot training CIFAR10_data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="train CIFAR10_data")

    # Plot test CIFAR10_data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="test CIFAR10_data")

    if predictions is not None:
        # Plot the predictions if they exist
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})
    plt.show()

plot_predictions()