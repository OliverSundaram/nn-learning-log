from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch import nn

# Classification is a problem of prediction whether something is one thing or another

# Make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

# Turn CIFAR10_data into tensors and create train and test splits
# Turn CIFAR10_data into tensors
X = torch.Tensor(X).type(torch.float)
y = torch.Tensor(y).type(torch.float)

# Split CIFAR10_data
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2, # 0.2 = 20% of CIFAR10_data will be test & 80% will be train
                                                    random_state=42)

# Setup device agnostic code to run on a GPU if there is one
device = "cuda" if torch.cuda.is_available() else "cpu"

X_train = torch.Tensor(X_train).to(device)
X_test = torch.Tensor(X_test).to(device)
y_train = torch.Tensor(y_train).to(device)
y_test = torch.Tensor(y_test).to(device)

# Create a model that:
# 1. Subclasses nn.Module
# 2. Create 2 nn.Linear() layers that are capable of handling the shapes of our CIFAR10_data
# 3. Defines a forward() method that outlines the forward pass (or forward computation) of the model
# 4. Instantiate an instance of our model class and send it to the target device

# Construct a module that subclasses nn.Module
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. Create 2 nn.Linear() layers
        self.layer_1 = nn.Linear(in_features=2, out_features=8) # Takes in 2 inputs and upscales to 8 neurons in a hidden layer
        self.layer_2 = nn.Linear(in_features=8, out_features=1) # Takes in 8 inputs from the hidden layer and outputs a single value

    # 3. Define a forward() method that outlines the forward pass
    def forward(self, x):
        return self.layer_2(self.layer_1(x))

# 4. Make an instance of the model class and send it ot the target device
# model_0 = CircleModelV1().to(device)

model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)

# Setup loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss() # Contains the sigmoid activation function built in
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

# Calculate accuracy - out of 100 examples what percentage does our model get right?
def accuracy_fn(y_true, y_preds):
    correct = torch.eq(y_true, y_preds).sum().item()
    acc = (correct/len(y_preds)) * 100
    return acc

# train model
# 1. Forward pass
# 2. Calculate the loss
# 3. Optimizer zero grad
# 4. Loss backward (backpropagation)
# 5. Optimizer step (gradient descent)

model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test)[:5]

# Pass the model the test CIFAR10_data, which returns values, then pass them through the sigmoid activation function to turn them into probabilities
y_pred_probs = torch.sigmoid(y_logits)

# Find the predicted labels
y_preds = torch.round(y_pred_probs)

# In full:
y_preds = torch.round(torch.sigmoid(model_0(X_test)[:5]))

