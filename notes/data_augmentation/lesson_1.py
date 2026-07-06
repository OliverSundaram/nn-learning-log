from torchvision import transforms
import torch
from pathlib import Path
from PIL import Image
import random
from matplotlib import pyplot as plt



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
            plt.show()



train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

image_path_list = list(image_path.glob("*/*/*.jpg"))

plot_transformed_images(
    image_paths=image_path_list,
    transform=train_transform,
    n=3,
    seed=None
)