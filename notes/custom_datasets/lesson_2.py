import torch
from torch import nn
import os
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

def walk_through_dir(dir_path):
    """Walks through dir_path returning its contents."""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
walk_through_dir(image_path)



train_dir = image_path / "train"
test_dir = image_path / "test"
print(train_dir, test_dir)