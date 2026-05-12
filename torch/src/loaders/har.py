import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, Dataset

class HARLoader:
    def __init__(self, batch_size: int, normalize, ltarget_transform, debug_level: bool):
        self.name = 'har'
        self.debug = debug_level > 1
        trainset = HARSubset("train", ltarget_transform)
        testset = HARSubset("test", ltarget_transform)
        validset = HARSubset("val", ltarget_transform)
        infset_indices = torch.randperm(len(trainset)).tolist()[:len(validset)]
        traininfset = Subset(trainset, infset_indices)

        # Select class to keep 
        num_workers = 0 if self.debug else 8
        self.batch_size = batch_size
        self.train = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers)
        self.test = DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
        self.valid = DataLoader(validset, batch_size=batch_size, num_workers=num_workers)
        self.train_inf = DataLoader(traininfset, batch_size=batch_size, shuffle=False, 
                                    num_workers=num_workers)


        self.in_chan = 1
        self.in_size = (1, 300)
        self.out_dim = 6

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
