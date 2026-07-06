import torch
from torch import nn

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader

from notes.helper_functions.helper_functions import accuracy_fn

from notes.helper_functions.loops import train, test

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_data = datasets.Flowers102(root="data/Flowers102_data",
                                 split="train",
                                 transform=train_transform,
                                 download=True)
test_data = datasets.Flowers102(root="data/Flowers102_data",
                                split="test",
                                transform=test_transform,
                                download=True)
class_labels = train_data.classes

BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
class Flowers102(nn.Module):
    def __init__(self, input_channels, hidden_units, output_labels):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=hidden_units,
                      kernel_size=KERNEL_SIZE,
                      stride=STRIDE,
                      padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=KERNEL_SIZE,
                      stride=STRIDE,
                      padding=PADDING),
            nn.ReLU(),
            nn.MaxPool2d(4)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=KERNEL_SIZE,
                      stride=STRIDE,
                      padding=PADDING),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=KERNEL_SIZE,
                      stride=STRIDE,
                      padding=PADDING),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*8*8,
                      out_features=output_labels)
        )

    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

model = Flowers102(input_channels=3, hidden_units=10, output_labels=102)
model.load_state_dict(torch.load(f="models_state_dict/CNN_for_Flowers102.pth"))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.1)

epochs = 5
for epoch in range(1, epochs + 1):

    train_loss, train_acc = train(model=model, loss_fn=loss_fn, optimizer=optimizer, accuracy_fn=accuracy_fn, train_dataloader=train_dataloader)
    test_loss, test_acc = test(model=model, loss_fn=loss_fn, accuracy_fn=accuracy_fn, test_dataloader=test_dataloader)

    print(f"Epoch: {epoch} | Train loss: {train_loss} | Train acc: {train_acc}% | Test loss: {test_loss} | Test acc: {test_acc}%")

# Saving PyTorch model
from pathlib import Path

# 1. Create model directory
MODEL_PATH = Path("models_state_dict")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "CNN_for_Flowers102.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)