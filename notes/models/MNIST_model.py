import torch
from torch import nn

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from notes.helper_functions.helper_functions import accuracy_fn

train_data = datasets.MNIST(root="data/MNIST_data",
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
test_data = datasets.MNIST(root="data/MNIST_data",
                           train=False,
                           transform=transforms.ToTensor(),
                           download=True)
class_names = train_data.class_to_idx

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=32,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=32,
                             shuffle=True)

class MNISTModel(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_inputs,
                 output_targets):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=hidden_inputs,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_inputs,
                      out_channels=hidden_inputs,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_inputs,
                      out_channels=hidden_inputs,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_inputs,
                      out_channels=hidden_inputs,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_inputs*7*7,
                      out_features=output_targets)
        )

    def forward(self, x):
        z = self.conv_block_1(x)
        z = self.conv_block_2(z)
        z = self.classifier(z)
        return z

model = MNISTModel(input_channels=1,
                   hidden_inputs=16,
                   output_targets=len(class_names))
model.load_state_dict(torch.load(f="models_state_dict/CNN_for_MNIST.pth"))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.001)

epochs = 50
for epoch in range(1, epochs + 1):
    # Training loop
    model.train()
    train_loss, train_acc = 0, 0
    for batch in train_dataloader:
        X, y = batch
        y_logits = model(X)
        y_preds = torch.argmax(y_logits, dim=1)
        loss = loss_fn(y_logits, y)
        acc = accuracy_fn(y, y_preds)
        train_loss += loss.item()
        train_acc += acc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    # Testing loop
    model.eval()
    with torch.inference_mode():
        test_loss, test_acc = 0, 0
        for batch in test_dataloader:
            X, y = batch
            y_logits = model(X)
            y_preds = torch.argmax(y_logits, dim=1)
            loss = loss_fn(y_logits, y)
            acc = accuracy_fn(y, y_preds)
            test_loss += loss
            test_acc += acc
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    print(f"Epoch: {epoch} | Train loss: {train_loss} | Train acc: {train_acc}% | Test loss: {test_loss} | Test acc: {test_acc}%")



# Saving PyTorch model
from pathlib import Path

# 1. Create model directory
MODEL_PATH = Path("models_state_dict")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "CNN_for_MNIST.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)