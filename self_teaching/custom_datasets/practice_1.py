import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import torch
from torch import nn

import torchvision
from torchvision import transforms
from torchmetrics.classification.accuracy import BinaryAccuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PneumoniaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_data = PneumoniaDataset("data/chest_xray/train", train_transform)
test_data = PneumoniaDataset("data/chest_xray/test", test_transform)

BATCH_SIZE = 32
SHUFFLE = True

train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

class PneumoniaModel(nn.Module):
    def __init__(self,
                 input_channels: int,
                 hidden_units: int,
                 output_labels: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*8*8, out_features=output_labels)
        )

    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

model = PneumoniaModel(3, 10, 1).to(device)
model.load_state_dict(torch.load("models_state_dict/Pneumonia_model.pth"))
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
accuracy_fn = BinaryAccuracy().to(device)

epochs = 25
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=int(epochs / 5), gamma=0.1)

for epoch in range(1, epochs + 1):
    train_loss, train_acc = 0, 0
    model.train()
    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device).type(torch.float)

        y_logits = model(X).squeeze()
        y_preds = torch.round(torch.sigmoid(y_logits))
        loss = loss_fn(y_logits, y)
        acc = accuracy_fn(y_preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += acc.item()

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    scheduler.step()

    model.eval()
    with torch.inference_mode():
        test_loss, test_acc = 0, 0
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device).type(torch.float)

            y_logits = model(X).squeeze()
            y_preds = torch.round(torch.sigmoid(y_logits))
            loss = loss_fn(y_logits, y)
            acc = accuracy_fn(y_preds, y)

            test_loss += loss.item()
            test_acc += acc.item()

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    print(f"Epoch: {epoch} | Train loss: {train_loss} | Train acc: {train_acc} | Test loss: {test_loss} | Test acc: {test_acc}")

# Saving PyTorch model
from pathlib import Path

# 1. Create model directory
MODEL_PATH = Path("models_state_dict")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "Pneumonia_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)