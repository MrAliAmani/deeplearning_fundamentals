import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, img_dir, label_file):
        super(ImageDataset, self).__init__()
        self.img_dir = img_dir
        self.labels = torch.tensor(np.load(label_file, allow_pickle=True))
        self.transforms = transforms.ToTensor()

    def __getitem__(self, idx):
        img_pth = os.path.join(self.img_dir, "img_{}.jpg".format(idx))
        img = Image.open(img_pth)
        img = self.transforms(img).flatten()
        label = self.labels[idx]
        return {"data":img, "label":label}

    def __len__(self):
        return len(self.labels)
