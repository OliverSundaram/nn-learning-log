import torch
from torch import nn

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt

from notes.helper_functions.helper_functions import accuracy_fn
from pathlib import Path

from torch.utils.data import DataLoader

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=train_transform)
test_data = datasets.ImageFolder(root=test_dir,
                                 transform=test_transform)

BATCH_SIZE = 16

train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True)


class Food3Model(nn.Module):
    def __init__(self, in_channels, hidden_units, out_labels):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=out_labels)
        )

    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

model = Food3Model(3, 10, 3)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                                lr=0.0001)

if __name__ == "__main__":
    train_losses, test_losses, epoch_list = [], [], []
    epochs = 100
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = 0, 0
        model.train()
        epoch_list.append(epoch)
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
        train_losses.append(train_loss)

        model.eval()
        with torch.inference_mode():
            test_loss, test_acc = 0, 0
            for batch in test_dataloader:
                X, y = batch
                y_logits = model(X)
                y_preds = torch.argmax(y_logits, dim=1)
                loss = loss_fn(y_logits, y)
                acc = accuracy_fn(y, y_preds)
                test_loss += loss.item()
                test_acc += acc
            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)
            test_losses.append(test_loss)
        print(f"Epoch: {epoch} | Train loss: {train_loss} | Train acc: {train_acc}% | Test loss: {test_loss} | Test acc: {test_acc}%")

    def plot_loss_curves(train_losses: list[float], test_losses: list[float], epoch_list: list[int]):


        plt.plot(epoch_list, train_losses, c="orange", label="Train loss")
        plt.plot(epoch_list, test_losses, c="blue", label="Test loss")
        plt.title("Loss curve for training and testing")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()

    plot_loss_curves(train_losses, test_losses, epoch_list)

    # Saving PyTorch model
    from pathlib import Path

    # 1. Create model directory
    MODEL_PATH = Path("models_state_dict")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # 2. Create model save path
    MODEL_NAME = "Food3_with_trivial_augment_CNN.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    # 3. Save the model state dict
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(),
               f=MODEL_SAVE_PATH)