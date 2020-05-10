import os
import pickle
from lib.cityscape_labels import labels
from lib.constants import no_of_classes
from PIL import Image
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import torchvision
import torch.utils.data
import numpy as np
from typing import List


class SegmentationDataset(torch.utils.data.Dataset):
    """
    Data loader for the Segmentation Dataset. If data loading is a bottleneck, 
    you may want to optimize this in for faster training. Possibilities include
    pre-loading all images and annotations into memory before training, so as 
    to limit delays due to disk reads.
    """

    classes: List[str] = []

    def __init__(self, split="train", data_dir: str = "./data", transforms=None, shared_transform=None):
        assert(split in ["train", "val", "test"])
        self.img_dir = os.path.join(data_dir, split)

        # Generate names for the classes
        self.classes = ["" for x in range(no_of_classes)]
        for label in labels:
            self.classes[label.trainId] = label.name
        self.classes[no_of_classes - 1] = "misc"

        self.n_classes = len(self.classes)
        self.split = split
        self.data_img = pickle.load(open(f'data/{split}_img_paths.pkl', 'rb'))
        self.data_trainId = pickle.load(
            open(f'data/{split}_trainId_label_paths.pkl', 'rb'))
        self.transforms = transforms
        self.shared_transform = shared_transform

    def __len__(self):
        return len(self.data_img)

    def __getitem__(self, index):
        img = np.copy(Image.open(self.data_img[index])).astype(np.uint8)
        gt = np.copy(Image.open(self.data_trainId[index])).astype(np.uint8)
        if self.shared_transform is not None:
            combined = self.shared_transform(image=img, mask=gt)
            img = combined["image"]
            gt = combined["mask"]
        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = ToTensor()(img)
        gt = torch.LongTensor(np.copy(gt)).unsqueeze(0)
        return img, gt
