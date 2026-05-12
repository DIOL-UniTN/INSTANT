import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from utils.fff import leaf_target_transform

from PIL import Image

class MNISTLoader:
    def __init__(self, batch_size: int, transform, aug, aug_prob:float, ltarget_transform, debug_level: bool):
        self.name = 'mnist'
        self.debug = debug_level > 1
        self.aug_prob = aug_prob
        train_transform = transforms.Compose(list(transform) + list(aug))
        transform = transforms.Compose(list(transform))

        trainset = MNISTSubset(train=True, transform=train_transform, debug_level=debug_level, ltarget_transform=ltarget_transform)
        trainset, validset = torch.utils.data.random_split(trainset, [0.9, 0.1])
        testset = MNISTSubset(train=False, transform=transform, debug_level=0, ltarget_transform=ltarget_transform)
        infset_indices = torch.randperm(len(trainset)).tolist()[:len(validset)]
        traininfset = Subset(trainset, infset_indices)

        num_workers = 0 if self.debug else 8
        self.train = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers)
        self.valid = DataLoader(validset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers)
        self.test = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers)

        self.batch_size = batch_size
        self.in_chan = 1
        self.in_size = (28, 28)
        self.out_dim = 10

class MNISTSubset(MNIST):
    def __init__(self, train: bool, transform, debug_level: bool, ltarget_transform):
        super().__init__(root='./data', train=train, download=True, transform=transform)
        self.debug_level = debug_level
        self.ltarget_transform = ltarget_transform

    def __len__(self):
        if self.debug_level == 3:
            return 20
        elif self.debug_level == 2:
            return 10000
        return len(self.data)

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.ltarget_transform is not None:
            leaf_target = self.ltarget_transform(target)
            return img, target, leaf_target
        return img, target

