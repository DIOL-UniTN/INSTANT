import shutil
import torch
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS 
from torch.utils.data import Subset, DataLoader
from utils.audio_proc import fix_audio_length
from torchaudio.datasets.utils import _load_waveform
from tqdm import tqdm
from pathlib import Path
from glob import glob
from tqdm import tqdm
import numpy as np
import logging

TINY_LABELS = [
    'yes', 
    'no', 
    'up', 
    'down', 
    'left', 
    'right', 
    'on', 
    'off', 
    'stop', 
    'go', 
    'noise'
]
MEDIUM_LABELS = [
    "down",
    "eight",
    "five",
    "follow",
    "four",
    "go",
    "left",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "wow",
    "yes",
    "zero",
    "noise",
]
FULL_LABELS = [
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
    "noise",
]

LABELS = [TINY_LABELS, MEDIUM_LABELS, FULL_LABELS]

class SCLoader:
    def __init__(self, batch_size:int, size:int, feature, sample_rate:int, debug_level:bool):
        self.name = 'sc'
        self.debug = debug_level > 1
        self.batch_size = batch_size
        self.labels = LABELS[size] #TODO
        self.duration = 1.0 # TODO
        self.sr = sample_rate # To be resampled
        self.out_dim = len(self.labels)

        trainset = SubsetSC('./data', 'training', feature, sample_rate, self.debug, self.labels)
        testset = SubsetSC('./data', 'testing', feature, sample_rate, self.debug, self.labels)
        validset = SubsetSC('./data', 'validation', feature, sample_rate, self.debug, self.labels)
        infset_indices = torch.randperm(len(trainset)).tolist()[:len(validset)]
        traininfset = Subset(trainset, infset_indices)

        num_workers = 0 if self.debug else 8
        self.train = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers)
        self.valid = DataLoader(validset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers)
        self.test = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers)
        self.train_inf = DataLoader(traininfset, batch_size=batch_size, shuffle=False, 
                                    num_workers=num_workers)

        if feature:
            out_shape = feature(torch.randn(1, int(self.sr*self.duration))).shape
            self.in_chan, self.in_size = out_shape[0], tuple(out_shape[1:])
            logging.info(f"Feature size: {self.in_size}")
        else:
            self.in_chan, self.in_size = 1, (1, int(self.sr*self.duration))
            logging.info("Raw data")

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, dataset_dir: str='./data', 
                 subset:str='training', feature=None, 
                 sample_rate=16000,
                 debug: bool = False,
                 labels: list[str] = TINY_LABELS):
        super().__init__(dataset_dir, download=True, subset=subset)
        self.labels = labels
        self.debug = debug
        self.sr = sample_rate
        self.duration = 1.0
        self.resampler = torchaudio.transforms.Resample(16000, self.sr)
        self._walker = [file for file in self._walker if file.split('/')[-2] in self.labels] 
        self.subset=subset
        self.feature = feature

    def __len__(self):
        if self.debug:
            return 10
        return int(len(self._walker))

    def label_to_target(self, word):
        return torch.tensor(self.labels.index(word))

    def __getitem__(self, idx):
        metadata = self.get_metadata(idx)
        waveform = _load_waveform(self._archive, metadata[0], metadata[1])
        waveform = self.resampler(waveform)
        waveform = fix_audio_length(waveform, t=self.duration, sr=self.sr) # TODO Handle randomness for test? How to handle?
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-10)
        feat = self.feature(waveform)
        target = self.label_to_target(metadata[2])
        return feat, target
