import os
from PIL import Image
from torch.utils.data import Dataset

class BrainTumor(Dataset):
    def __init__(self,
                 root_dir: str | os.PathLike,
                 train: bool = True,
                 transform=None):
        self.root_dir = os.path.join(root_dir, "train" if train else "test")
        self.transform = transform
        self.classes = os.listdir(self.root_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for img_dir in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, img_dir), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        img = Image.open(img).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label