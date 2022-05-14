import os
import json
import math

import torch

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, pad, normalize
from PIL import Image


class COCODataset(Dataset):
    """
    Dataset to load data in COCO-style format and provide samples corresponding to individual objects.
    Images are cropped to the objects bounding box.
    """

    def __init__(self, annotations_file, image_dir, image_size, normalize_means=None, normalize_stds=None):
        """
        Args:
            annotations_file: path to COCO-style annotation file (.json)
            image_dir: path to the image folder
            image_size: desired size of the sample images, either a tuple (w,h) or an int if w=h
            normalize_means: List of means for each channel. Set None to disable normalization.
            normalize_stds: List of standard deviations for each channel. Set None to disable normalization.

        Items are returned in the format: (image, label) where label is an index in [0, NUM_CLASSES - 1].
        """
        
        self.image_dir = image_dir
        self.image_size = image_size if type(image_size) == tuple else (image_size, image_size)

        with open(annotations_file) as f:
            self.coco_dict = json.load(f)

        self.annotations = self.coco_dict["annotations"]
        
        self.id2file = {}
        for i in self.coco_dict["images"]:
            self.id2file[i["id"]] = os.path.join(image_dir, i["file_name"])

        self.id2label = {} # maps label id to label name
        self.idx2label = {} # maps index in 1-hot encoding to label name
        self.id2idx = {} # maps label id to index in 1-hot encoding
        for idx, i in enumerate(self.coco_dict["categories"]):
            self.id2label[i["id"]] = i["name"]
            self.idx2label[idx] = i["name"]
            self.id2idx[i["id"]] = idx
        
        self.NUM_CLASSES = len(self.id2label)

        if normalize_means is not None and normalize_stds is not None:
            self.normalize_means = normalize_means
            self.normalize_stds = normalize_stds
            self.normalize = True
        else:
            self.normalize = False

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        # load image
        image = Image.open(self.id2file[annotation["image_id"]])
        image = image.convert("RGB")
        # crop to bounding box
        xmin, ymin, w, h = annotation["bbox"]
        image = image.crop((int(xmin), int(ymin), int(xmin + w), int(ymin + h)))

        # resize
        image = image.resize(self.image_size)

        # convert to tensor and normalize
        image = to_tensor(image)

        if self.normalize:
            image = normalize(image, self.normalize_means, self.normalize_stds)

        label = torch.tensor(self.id2idx[annotation["category_id"]])

        return image, label


class COCODatasetWithID(COCODataset):
    """
    Provides the same functionality as COCODataset but in addition the id of the annotation is also returned.
    This can be useful for in-depth analysis of the results.
    """

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        annotation_id = self.annotations[idx]["id"]
        return image, label, annotation_id