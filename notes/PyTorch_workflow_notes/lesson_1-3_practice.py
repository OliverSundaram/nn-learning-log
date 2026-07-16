import torch
from torch import nn
import matplotlib.pyplot as plt
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

input_schematic = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

def gen_data(size: int = 100) -> torch.Tensor:
    data = []
    for i in range(size):
        data.append(random.choice(input_schematic))
    return torch.tensor(data, device=device)

def plot_loss_curve(epochs : list[int], train_loss_values : list[float], test_loss_values : list[float]) -> None:
    plt.plot(epochs, train_loss_values, label="Train loss")
    plt.plot(epochs, test_loss_values, label="Test loss")
    plt.legend()
    plt.title("train and test loss curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

# Set weights and biases that model will try to re-create
weights = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
bias = torch.tensor(0.0, device=device)

# Generate the CIFAR10_data to give the model, and the simulated perfect outputs that the model should try to get
data = gen_data(100)
outputs = torch.sum(data * weights, dim=1) + bias

# Separate CIFAR10_data into training and testing
train_split = int(len(data) * 0.8)

train_data = data[:train_split]
train_outputs = outputs[:train_split]
test_data = data[train_split:]
test_outputs = outputs[train_split:]

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=4,
                                      out_features=1,
                                      device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x).squeeze()

model = LinearRegressionModel()
model.to("cuda")
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.001)

# Print model information before training
print(f"Before training | {model.state_dict()}")

epoch_count = []
train_loss_values = []
test_loss_values = []

epochs = 10000

for epoch in range(1, epochs + 1):
    model.train()

    output_predictions = model(train_data)

    train_loss = loss_fn(output_predictions, train_outputs)

    optimizer.zero_grad()

    train_loss.backward()

    optimizer.step()

    if epoch % (epochs / 5) == 0 or epoch == 1:
        model.eval()

        with torch.inference_mode():
            test_predictions = model(test_data)
            test_loss = loss_fn(test_predictions, test_outputs)

        print(f"Test at epoch {epoch} | train loss : {train_loss.item()} | Testing loss : {test_loss.item()}")

        epoch_count.append(epoch)
        train_loss_values.append(train_loss.item())
        test_loss_values.append(test_loss.item())

print(f"After training | {model.state_dict()}")
plot_loss_curve(epoch_count, train_loss_values, test_loss_values)


from pathlib import Path
# 1. Create models_state_dict directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "model_1_4_input_0_hidden_1_output.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)


# Create a new instance of model
model = LinearRegressionModel()
model.load_state_dict(torch.load("models/model_1_4_input_0_hidden_1_output.pth"))
model.to(device=device)

print(model.state_dict())