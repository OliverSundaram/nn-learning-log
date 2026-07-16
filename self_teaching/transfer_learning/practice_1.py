import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import torch
from torch import nn

import torchvision
from torchvision import transforms, models
from torchmetrics.classification.accuracy import BinaryAccuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class AnimalsDataset(Dataset):
    def __init__(self, root_dir: str, train: bool = True, transform: transforms.Compose | None = None):
        """
        Dataset with 2 classes: horse and lion
        :param root_dir: The path to the directory of where all the data is stored
        :param train: Determines if data is from train or test
        :param transform: Transform to apply to data
        """
        self.root_dir = os.path.join(root_dir, "train" if train else "test")
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for img in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, img), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label = self.samples[idx]
        image = Image.open(image).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label



train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = AnimalsDataset("data/animals", True, train_transform)
test_data = AnimalsDataset("data/animals", False, test_transform)
class_names = train_data.classes

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
num_ftrs = model.fc.in_features

model.fc = nn.Linear(in_features=num_ftrs, out_features=1)
model.to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.fc.parameters(), lr=0.01)
accuracy_fn = BinaryAccuracy().to(device)
epochs = 25
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(epochs / 5), 0.1)



for epoch in range(1, epochs + 1):
    train_loss, train_acc = 0, 0
    model.train()
    for X, y in train_dataloader:
        accuracy_fn.reset()
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
MODEL_NAME = "Animals.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)
