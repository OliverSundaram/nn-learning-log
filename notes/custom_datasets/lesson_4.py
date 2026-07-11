import matplotlib.pyplot as plt
import torch
import torchvision.io
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import random
from pathlib import Path
from PIL import Image
from notes.helper_functions.helper_functions import accuracy_fn

# Transform for image
data_transform = transforms.Compose([
    # Resize images to 64x64
    transforms.Resize((64, 64)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(),
    # Turn image to a torch tensor
    transforms.ToTensor()
])

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

image_path_list = list(image_path.glob("*/*/*.jpg"))

# Pick random image path
random_image_path = random.choice(image_path_list)

# Get image class name
image_class = random_image_path.parent.stem

# Open image
img = Image.open(random_image_path)

transformed_image = data_transform(img)

def plot_transformed_images(image_paths: list, transform, n=3, seed=None):
    """
    Selects random images from a pth of images and loads/transforms them then plots the original vs the transformed version.
    :param image_paths:
    :param transform:
    :param n: Number of random images
    :param seed: Random number to set random seed for reproducibility
    :return: None
    """
    if seed:
        random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize: {f.size}")
            ax[0].axis(False)

            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed\nShape: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

from torchvision import datasets
train_dir = image_path / "train"
test_dir = image_path / "test"

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

class_names = train_data.class_to_idx

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=16,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=16,
                             shuffle=True)

class Food3Model(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_units,
                 output_labels):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
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
            nn.MaxPool2d(4)
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
            nn.Linear(in_features=hidden_units*8*8,
                      out_features=output_labels)
        )

    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

model = Food3Model(3, 10, 3)
model.load_state_dict(torch.load(f="models_state_dict/CNN_for_Food3.pth"))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.0000001)

epochs = 50
for epoch in range(1, epochs + 1):
    train_loss, train_acc = 0, 0
    model.train()
    for batch in train_dataloader:
        X, y = batch
        y_logits = model(X)
        y_preds = torch.argmax(y_logits, dim=1)
        loss = loss_fn(y_logits, y)
        acc = accuracy_fn(y, y_preds)
        train_loss += loss
        train_acc += acc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    # Testing
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
    print(f"Epoch: {epoch} | Train loss: {train_loss} | Train acc: {train_acc} | Test loss: {test_loss} | Test acc: {test_acc}")



# Saving PyTorch model
from pathlib import Path

# 1. Create model directory
MODEL_PATH = Path("models_state_dict")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "CNN_for_Food3.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)