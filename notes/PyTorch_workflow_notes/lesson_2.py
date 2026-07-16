import torch
from torch import nn
import matplotlib.pyplot as plt
import lesson_1

# PyTorch model building essentials
# torch.nn - contains building blocks for neural networks
# torch.nn.Parameter - what parameters should our model try to learn
# torch.nn.Module - The base class for all neural network modules, if you subclass it, you should overwrite forward
# torch.optim - Where the optimisers are in PyTorch, help with gradient descent
# def forward() - All nn.Module subclasses require you to overwrite forward, this method defines what happens in the forward computation

# Create known parameters
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(1)
y = weight * X + bias

# Splitting CIFAR10_data into training and test sets
# Create a train/test split

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# 2. Building model
class LinearRegressionModel(nn.Module): # almost everything from PyTorch inherits from nn.Module
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))


    # Forward method to define computation in model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # "x" is the input CIFAR10_data
        return self.weights * x + self.bias # Linear regression formula

torch.manual_seed(42)

# Create an instance of the model
model_0 = LinearRegressionModel()



# 3. Train model
# Moving the random parameters to parameters that better represent the CIFAR10_data

# One way to measure how poor or how wrong your models_state_dict predictions are is to use a loss function

# Things we need to train:
# Loss function - A function to measure how wrong your model's predictions are to the ideal outputs, lower is better
# Optimizer: Takes into account the loss of a model and adjusts the model's parameters to improve the loss function



# Set up a loss function
loss_fn = nn.L1Loss()

# Setup an optimizer (stochastic gradient descent)
optimizer = torch.optim.SGD(params=model_0.parameters(), # The models_state_dict parameters that you would like to optimize
                            lr=0.0001) # Learning rate - how big or small the optimizer changes the parameters



# Building a training loop and testing loop in PyTorch
# things you need in a training loop:
# 0. Loop through the CIFAR10_data
# 1. Forward pass (This involves CIFAR10_data moving through the model's forward()) to
#    make predictions on CIFAR10_data - also called forward propagation
# 2. Calculate the loss (compare forward pass predictions to the ground truth labels)
# 3. Optimizer zero grad
# 4. Loss backward - move backward through the network to calculate the
#    gradients of each of the parameters of our model with respect to the loss (backpropagation)
# 5. Optimizer step - use the optimizer to adjust our model's
#    parameters to try and improve the loss (gradient descent)

# train loop
epochs = 100000

print("Before training:", model_0.state_dict())

for epoch in range(epochs):
    # Set the model to training mode
    model_0.train()

    # 1. Forward pass
    y_pred = model_0(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train)

    # 3. Zero optimizer gradient (they accumulate over time)
    optimizer.zero_grad()

    # 4. Perform backpropagation on the loss with respect to the parameters of the model
    loss.backward()

    # 5. Step the optimizer (perform gradient descent)
    optimizer.step()

    model_0.eval() # turns off different settings in the model not needed for evaluation/testing



print("After training:", model_0.state_dict())