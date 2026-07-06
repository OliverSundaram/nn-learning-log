import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

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

test_dir = image_path / "test"

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

class_names = test_data.classes

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

while True:
     for image in test_data:
         X, y = image
         X = X.unsqueeze(dim=0)
         y_logits = model(X)
         y_pred = torch.argmax(y_logits, dim=1)

         X = X.squeeze()
         X = torch.permute(X, dims=(1, 2, 0))

         plt.imshow(X)
         plt.axis(False)
         print(f"Label: {class_names[y]} | AI guess: {class_names[y_pred.item()]}")
         plt.show()