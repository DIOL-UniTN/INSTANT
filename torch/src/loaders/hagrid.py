import json
import logging
import os
from typing import Dict, List, Tuple, Optional
import pickle

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm
from pathlib import Path

from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import DataLoader
import albumentations

SET_SIZE = 1086225
DATA_DIR = "data/hagrid"

SETS = ["train", "val", "test"]
IMAGES = (".jpeg", ".jpg", ".jp2", ".png", ".tiff", ".jfif", ".bmp", ".webp", ".heic")

targets = {
    0: "grabbing",
    1: "grip",
    2: "holy",
    3: "point",
    4: "call",
    5: "three3",
    6: "timeout",
    7: "xsign",
    8: "hand_heart",
    9: "hand_heart2",
    10: "little_finger",
    11: "middle_finger",
    12: "take_picture",
    13: "dislike",
    14: "fist",
    15: "four",
    16: "like",
    17: "mute",
    18: "ok",
    19: "one",
    20: "palm",
    21: "peace",
    22: "peace_inverted",
    23: "rock",
    24: "stop",
    25: "stop_inverted",
    26: "three",
    27: "three2",
    28: "two_up",
    29: "two_up_inverted",
    30: "three_gun",
    31: "thumb_index",
    32: "thumb_index2",
    33: "no_gesture",
}

class HAGRIDLoader:
    def __init__(self, batch_size: int, debug_level: bool, 
                 transforms: List[albumentations.BasicTransform]):
        self.name = 'hagrid'
        self.debug = debug_level > 1
        self.data_dir = Path(DATA_DIR)

        self.transforms = albumentations.Compose(transforms)
        trainset = GestureDataset(targets, "train", self.transforms)
        validset = GestureDataset(targets, "val", self.transforms)
        testset = GestureDataset(targets, "test", self.transforms)

        num_workers = 0 if self.debug else 8
        self.train = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers)
        self.valid = DataLoader(validset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers)
        self.test = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers)

        self.batch_size = batch_size
        self.in_chan = 3
        self.in_size = (224, 224)
        self.out_dim = 34

class HagridDataset(Dataset):
    """
    Custom Dataset for HaGRID
    """

    def __init__(self, targets: Dict[int, str], dataset_type: str, transforms):
        """
        Parameters
        ----------
        conf : DictConfig
        Config for dataset
        dataset_type : str
        Type of dataset
        """
        self.labels =  {v: k for k, v in targets.items()}

        self.dataset_type = dataset_type

        subset = None if dataset_type == "train" else -1

        self.main_path = Path(DATA_DIR)
        if (self.main_path / f"{dataset_type}_annotations.pkl").exists():
            self.annotations = pd.read_pickle(self.main_path / f"{dataset_type}_annotations.pkl")
        else:
            self.annotations = self.__read_annotations(subset)
            self.annotations.to_pickle(self.main_path / f"{dataset_type}_annotations.pkl")
        self.transforms = transforms

        # Process for later
        current_set_size = sum(1 for p in (self.main_path/"data_processed").rglob("*") if p.is_file())
        print(current_set_size)
        if current_set_size < SET_SIZE:
            self.process_set()

    @staticmethod
    def _load_image(image_path: str):
        """
        Load image from path

        Parameters
        ----------
        image_path : str
        Path to image
        """
        image = Image.open(image_path).convert("RGB")

        return image

    def process_set(self) -> None:
        logging.info(f"Processing {self.dataset_type} set")
        for i in tqdm(range(len(self))):
            row = self.annotations.iloc[[i]].to_dict("records")[0]

            image_pth = self.main_path/"data"/row["target"]/row["name"]

            try:
                image = self._load_image(image_pth)
                image = self.transforms(image=np.array(image))["image"]
                filename = os.path.splitext(row['name'])[0]
                processed_image_pth = self.main_path/"data_processed"/row['target']
                Path(processed_image_pth).mkdir(parents=True, exist_ok=True)
                image = torch.from_numpy(image).permute(2, 0, 1)
                torch.save(image, processed_image_pth/f"{filename}.pt")
            except:
                print(f"Error opening image")


    @staticmethod
    def __get_files_from_dir(pth: str, extns: Tuple) -> List:
        """
        Get list of files from dir according to extensions(extns)

        Parameters
        ----------
        pth : str
        Path ot dir
        extns: Tuple
        Set of file extensions
        """
        if not os.path.exists(pth):
            logging.warning(f"Dataset directory doesn't exist {pth}")
            return []
        files = [f for f in os.listdir(pth) if f.endswith(extns)]
        return files

    def __read_annotations(self, subset: Optional[int] = None) -> pd.DataFrame:
        """
        Read annotations json

        Parameters
        ----------
        subset : int
        Length of subset for each target

        Returns
        -------
        pd.DataFrame
        Dataframe with annotations
        """
        exists_images = set()
        annotations_all = []

        for _, target in tqdm(targets.items(), desc=f"Prepare {self.dataset_type} dataset"):
            target_tsv = os.path.join(self.main_path/"annotations"/self.dataset_type, f"{target}.json")
            if os.path.exists(target_tsv):
                with open(target_tsv, "r") as file:
                    json_annotation = json.load(file)

                json_annotation = [
                    {**annotation, "name": f"{name}.jpg"} for name, annotation in json_annotation.items()
                ]
                if subset != -1 and subset is not None:
                    json_annotation = json_annotation[:subset]

                annotation = pd.DataFrame(json_annotation)
                annotation["target"] = target
                annotations_all.append(annotation)
                exists_images.update(self.__get_files_from_dir(os.path.join(self.main_path/"data", target), IMAGES))
            else:
                logging.info(f"Database for {target} not found")

        annotations_all = pd.concat(annotations_all, ignore_index=True)
        annotations_all["exists"] = annotations_all["name"].isin(exists_images)
        return annotations_all[annotations_all["exists"]]

    def __getitem__(self, item):
        """
        Get item from annotations
        """
        raise NotImplementedError

    def __len__(self):
        """
        Get length of dataset
        """
        return self.annotations.shape[0]


class GestureDataset(HagridDataset):
    def __init__(self, 
                 targets: dict, 
                 dataset_type: str, 
                 transforms: albumentations.Compose
                 ):
        """
        Parameters
        ----------
        conf : DictConfig
        Config for dataset
        dataset_type : str
        Type of dataset
        """
        super().__init__(targets, dataset_type, transforms)
        self.annotations = self.annotations[
            ~self.annotations.apply(lambda x: x["labels"] == ["no_gesture"] and x["target"] != "no_gesture", axis=1)
        ]

        self.dataset_type = dataset_type

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item from annotations

        Parameters
        ----------
        index : int
        Index of annotation item

        Returns
        -------
        Tuple[Image.Image, Dict]
        Image and target
        """
        row = self.annotations.iloc[[index]].to_dict("records")[0]

        image_pth = (self.main_path/"data_processed"/row["target"]/row["name"]).with_suffix(".pt")
        image = torch.load(image_pth)

        labels = row["labels"]
        gesture = ""
        if row["target"] == "no_gesture":
            gesture = "no_gesture"
        else:
            for label in labels:
                if label == "no_gesture":
                    continue
                else:
                    gesture = label
        label = torch.tensor(self.labels[gesture])
        return image, label
