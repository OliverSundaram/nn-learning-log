import random
from PIL import Image
import torch
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

image_path_list = list(image_path.glob("*/*/*.jpg"))

# Pick random image path
random_image_path = random.choice(image_path_list)

# Get image class name
image_class = random_image_path.parent.stem

# Open image
img = Image.open(random_image_path)

print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")

# turn the image into an array
img_as_array = np.asarray(img)
plt.imshow(img_as_array)
plt.title(image_class)
plt.axis(False)
plt.show()