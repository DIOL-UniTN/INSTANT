import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader

from PIL import Image

SUBSET = [0, 6, 9]

class MNISTSubsetLoader:
    def __init__(self, batch_size: int, transform, aug, aug_prob:float, debug_level: bool):
        self.name = 'mnist'
        self.debug_level = debug_level
        self.aug_prob = aug_prob
        train_transform = transforms.Compose(list(transform) + list(aug))
        transform = transforms.Compose(list(transform))

        trainset = MNISTSubset(train=True, transform=train_transform, debug_level=self.debug_level)
        validset = MNISTSubset(train=True, transform=transform, debug_level=self.debug_level)
        testset = MNISTSubset(train=False, transform=transform, debug_level=0)

        whole_range = torch.randperm(len(trainset))
        val_len = int(len(whole_range)*0.1)
        train_len = len(whole_range)-val_len

        trainset = Subset(trainset, whole_range[:train_len])
        validset = Subset(validset, whole_range[val_len:])

        num_workers = 0 if not (self.debug_level == 3) else 8
        self.train = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.valid = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.batch_size = batch_size

        self.in_chan = 1
        self.in_size = (28, 28)
        self.out_dim = 10

class MNISTSubset(MNIST):
    def __init__(self, train: bool, transform, debug_level: bool):
        super().__init__(root='./data', train=train, download=True, transform=transform)
        self.debug_level = debug_level
        self.data = [(img, target) for (img, target) in zip(self.data, self.targets) 
                                   if target in SUBSET]

    def __len__(self):
        if self.debug_level == 3:
            return 2
        elif self.debug_level == 2:
            return 10000
        return len(self.data)

    def __getitem__(self, index: int):
        img, target = self.data[index][0], int(self.data[index][1])

        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

