import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

custom_image_path = "data/slice_of_pizza.png"

custom_image_uint8 = torchvision.io.read_image(str(custom_image_path)) / 255

image_transform = transforms.Compose([
    transforms.Resize((64, 64))
])

image = image_transform(custom_image_uint8)
plt.imshow(image.permute(1, 2, 0))
plt.show()

from notes.data_augmentation.lesson_3 import Food3Model
model = Food3Model(3, 10, 3)
model.load_state_dict(torch.load("models_state_dict/Food3_with_trivial_augment_CNN.pth"))

y_logits = model(image.unsqueeze(0))
y_pred = torch.argmax(y_logits, dim=1)
print(y_pred)