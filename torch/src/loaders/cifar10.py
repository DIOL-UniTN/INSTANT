from pathlib import Path
import logging
import numpy as np
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, TensorDataset

DATA_DIR = "data/cifar-10-batches-py"
SETS = ["train", "valid", "test"]

class CIFAR10Loader:
    def __init__(self, batch_size: int, num_workers: int, normalize):
        self.name = 'CIFAR10'
        self.data_dir = Path(DATA_DIR)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        transform = transforms.Compose([transforms.ToTensor(), normalize])

        full_dataset = CIFAR10(root=self.data_dir.parent, train=True, download=True,
                             transform=transform)
        num_train = int(0.9 * len(full_dataset))
        indices = np.random.permutation(len(full_dataset))
        train_indices, val_indices = indices[:num_train], indices[num_train:]
        train_dataset = CIFAR10(root='data', train=True, transform=transform)
        val_dataset = CIFAR10(root='data', train=True, transform=transform)

        trainset = Subset(train_dataset, train_indices)
        validset = Subset(val_dataset, val_indices)
        testset = CIFAR10(root=self.data_dir.parent, train=False, download=True, 
                          transform=transform)

        tensor_missing = sum([(not (self.data_dir/f"{dset}_tensors.pt").exists() 
                               or not (self.data_dir/f"{dset}_labels.pt").exists())
                              for dset in SETS])
        if tensor_missing:
            self.get_tensors([trainset, validset, testset])

        train_tensors, train_labels = (torch.load(self.data_dir/"train_tensors.pt"), 
                                       torch.load(self.data_dir/"train_labels.pt"))
        valid_tensors, valid_labels = (torch.load(self.data_dir/"valid_tensors.pt"), 
                                       torch.load(self.data_dir/"valid_labels.pt"))
        test_tensors, test_labels = (torch.load(self.data_dir/"test_tensors.pt"), 
                                       torch.load(self.data_dir/"test_labels.pt"))
        trains = TrainTensorDataset(train_tensors, train_labels, transform=transform_train)
        valids = TensorDataset(valid_tensors, valid_labels)
        tests = TensorDataset(test_tensors, test_labels)

        self.train = DataLoader(trains, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers)
        self.valid = DataLoader(valids, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers)
        self.test = DataLoader(tests, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers)

        self.batch_size = batch_size

        self.in_chan = 3
        self.in_size = (32, 32)
        self.out_dim = 10

    def get_tensors(self, sets):
        logging.info(f"Saving {self.name} tensors...")
        for i, dset in enumerate(SETS):
            images, labels = [], []
            for img, label in sets[i]:
                images.append(img)
                labels.append(torch.tensor(label))
            torch.save(torch.stack(images), self.data_dir/f"{dset}_tensors.pt") 
            torch.save(torch.stack(labels), self.data_dir/f"{dset}_labels.pt")

    def get_config(self):
        return {
            "task": self.name,
            "in_chan": self.in_chan,
            "in_size": self.in_size,
            "out_dim": self.out_dim,
            "train_samples": len(self.train.dataset),
            "valid_samples": len(self.valid.dataset),
            "test_samples": len(self.test.dataset),
        }

class TrainTensorDataset(TensorDataset):
    def __init__(self, *tensors, transform=None):
        super().__init__(*tensors)
        self.transform = transform

    def __getitem__(self, index):
        data = super().__getitem__(index)
        if self.transform:
            # If your dataset returns (data, target), apply transform to data only
            data_transformed = self.transform(data[0])
            return (data_transformed, *data[1:])
        return data
