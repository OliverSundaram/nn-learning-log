import torch
from torch import nn, optim, device, cuda
from torch.utils.data import DataLoader
from torch.cuda import is_available
from pathlib import Path

from datasets import BrainTumor
from loops import train, test
from model import model

from torchmetrics.classification import MulticlassAccuracy
from torchvision import transforms

device = device("cuda" if is_available() else "cpu")
print(f"Device in use: {device}")

BATCH_SIZE = 32
EPOCHS = 15
SCHEDULER_STEP = 5

torch.manual_seed(42)
if is_available():
    cuda.manual_seed(42)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_data = BrainTumor("data", True, train_transform)
test_data = BrainTumor("data", False, test_transform)
class_names = train_data.classes

train_dataloader = DataLoader(train_data, BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, BATCH_SIZE, shuffle=False)

model.to(device)

glioma_idx = class_names.index("glioma")
class_weights = torch.ones(len(class_names))
class_weights[glioma_idx] = 2
class_weights = class_weights.to(device)

optimizer = optim.AdamW(params=model.parameters(), lr=1e-4, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
acc_fn = MulticlassAccuracy(num_classes=4).to(device)
scheduler = optim.lr_scheduler.StepLR(optimizer, SCHEDULER_STEP, gamma=0.1)

best_test_acc = 0.0
best_path = Path("brain_tumor_model.pth")

for epoch in range(1, EPOCHS + 1):
    model, train_loss, train_acc = train(model=model, device=device, loss_fn=loss_fn, optimizer=optimizer, acc_fn=acc_fn, train_dataloader=train_dataloader)
    test_loss, test_acc = test(model=model, device=device, loss_fn=loss_fn, acc_fn=acc_fn, test_dataloader=test_dataloader)
    scheduler.step()

    print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {round(train_acc * 100, 4)}% | Test loss: {test_loss:.4f} | Test acc: {round(test_acc * 100, 4)}%")

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(obj=model.state_dict(), f=best_path)
        print(f"  New best test acc: {round(best_test_acc * 100, 4)}% — saved to {best_path}")

print(f"\nTraining complete. Best test accuracy: {round(best_test_acc * 100, 4)}%")