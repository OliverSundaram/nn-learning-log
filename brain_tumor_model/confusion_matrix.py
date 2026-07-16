import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from datasets import BrainTumor
from model import model
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_data = BrainTumor("data", False, test_transform)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)
class_names = test_data.classes

model.load_state_dict(torch.load("brain_tumor_params_resnet18_2.pth"))
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.inference_mode():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        preds = torch.argmax(model(images), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()