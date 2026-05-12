import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision.datasets import ImageFolder
from pathlib import Path

DATA_DIR = "data/tiny-224"

class TinyImageNetLoader:
    def __init__(self, batch_size: int, num_workers: int, normalize):
        self.name = 'tinyimagenet'
        self.data_dir = Path(DATA_DIR)
        transforms_train = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transforms_eval = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        trainset = ImageFolder(str(self.data_dir/"train"), transforms_train)
        validset = ImageFolder(str(self.data_dir/"val"), transforms_eval)
        testset = ImageFolder(str(self.data_dir/"test"), transforms_eval)

        num_workers = 8
        self.batch_size = batch_size
        self.train = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers)
        self.test = DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
        self.valid = DataLoader(validset, batch_size=batch_size, num_workers=num_workers)

        self.in_chan = 3
        self.in_size = (224, 224)
        self.out_dim = 200

class HARSubset(Dataset):
    def __init__(self, fold="train", ltarget_transform=None):
        self.data = pickle.load(open(f"data/har/{fold}_data.summary", "rb"), encoding="latin1")
        self.labels = pickle.load(open(f"data/har/{fold}_labels.summary", "rb"), encoding="latin1")
        self.ltarget_transform = ltarget_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        signal = self.data[index]
        target = self.labels[index]
        target = np.argmax(target, axis=0)
        if self.ltarget_transform is not None:
            leaf_target = self.ltarget_transform(target)
            return signal, target, leaf_target
        return signal, target
