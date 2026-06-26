import torch
from torch import nn
import matplotlib.pyplot as plt

weights = torch.tensor([0.3, 0.7])
bias = 1.2

def gen_data(size: int = 1) -> torch.Tensor:
    if size < 1:
        return torch.tensor([])
    total_inputs = size * 2
    inputs = torch.arange(0, total_inputs) / 100
    return inputs.reshape(size, 2)

inputs = gen_data(100)
outputs = torch.sum(inputs * weights, dim=1) + bias

# Split data into training and testing
train_split = int(0.8 * len(inputs))
train_inputs, test_inputs = inputs[:train_split], inputs[train_split:]
train_outputs, test_outputs = outputs[:train_split], outputs[train_split:]

def plot_predictions(train_inputs=train_inputs,
                     train_outputs=train_outputs,
                     test_inputs=test_inputs,
                     test_outputs=test_outputs,
                     predictions=None) -> None:

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(train_inputs[:, 0], train_inputs[:, 1], train_outputs, c="b", s=10, label="Training data") # Plot training data in blue
    ax.scatter(test_inputs[:, 0], test_inputs[:, 1], test_outputs, c="g", s=10, label="Testing data") # Plot test data in green

    if predictions is not None:
        ax.scatter(test_inputs[:, 0], test_inputs[:, 1], predictions.detach().cpu().numpy(), c="r", s=4, label="Predictions") # Plot the predictions if they exist in red

    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    ax.set_zlabel("Output")
    ax.set_title("3D Data and Predictions Visualization")

    ax.legend(prop={"size": 12})
    plt.show()


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(2,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weights) + self.bias

model_0 = LinearRegressionModel()
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.00001) # Learning rate - how big or small the optimizer changes the parameters

epochs = 300000

print("Before training:", model_0.state_dict())
# Input count
i = 0
for epoch in range(epochs):
    model_0.train()

    output_preds = model_0(train_inputs)

    loss = loss_fn(output_preds, train_outputs)

    # Zero optimizer gradient (they accumulate over time)
    optimizer.zero_grad()

    # Perform backpropagation on the loss with respect to the parameters of the model
    loss.backward()

    # Step the optimizer (perform gradient descent)
    optimizer.step()

    if epoch % 10000 == 0:
        print(f"Epoch: {epoch} | Loss: {loss.item()}")


print("After training:", model_0.state_dict())
model_0.eval()
with torch.inference_mode():
    test_preds = model_0(test_inputs)
plot_predictions(predictions=test_preds)