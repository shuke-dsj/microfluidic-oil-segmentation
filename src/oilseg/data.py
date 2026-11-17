"""Fail-fast image/mask pairing and paired spatial augmentation."""
from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from .labels import read_mask

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def image_files(directory):
    paths = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f'No images found: {directory}')
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError(f'Duplicate image stems: {directory}')
    return paths


def paired_files(images, masks):
    images, masks = image_files(images), image_files(masks)
    labels = {p.stem: p for p in masks}
    if {p.stem for p in images} != set(labels):
        raise ValueError('Image/mask stems must match exactly; positional pairing is disabled')
    return [(p, labels[p.stem]) for p in images]


class OilDataset(Dataset):
    def __init__(self, pairs, training=False, crop_size=512):
        self.pairs, self.training, self.crop_size = pairs, training, crop_size
        self.jitter = transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.1, hue=.05)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as im:
            image = im.convert('RGB')
        mask = read_mask(mask_path)
        if image.size != (mask.shape[1], mask.shape[0]):
            raise ValueError(f'Image/mask size mismatch: {image_path.name}')
        target = Image.fromarray(mask.astype(np.uint8))
        if self.training:
            w,h = image.size
            padding = (0, 0, max(0,self.crop_size-w), max(0,self.crop_size-h))
            if any(padding):
                image = TF.pad(image,padding,padding_mode='edge')
                target = ImageOps.expand(target, border=padding, fill=0)
            i,j,h,w = transforms.RandomCrop.get_params(image,(self.crop_size,self.crop_size))
            image,target = TF.crop(image,i,j,h,w),TF.crop(target,i,j,h,w)
            for transform in (TF.hflip,TF.vflip):
                if random.random() > .5:
                    image,target = transform(image),transform(target)
            image = self.jitter(image)
        return TF.to_tensor(image),torch.from_numpy(np.array(target,dtype=np.int64))
