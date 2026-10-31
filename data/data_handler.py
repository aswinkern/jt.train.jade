import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


# Normalized black pixel value (RGB 0,0,0 after DINOv2 normalization)
NORM_BLACK = [-2.1179, -2.0357, -1.8044]


class AnomalyAugmentation:
    """
    Augmentation pipeline for anomaly detection training.
    HSV shift, Gaussian blur, and optional random window mask.
    """
    def __init__(self, image_size, aug_config):
        self.image_size = image_size
        self.aug = aug_config

    def __call__(self, image):
        image = TF.to_tensor(image)

        # HSV Shift
        if self.aug.get("hsv_shift", True):
            image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
            image = TF.adjust_contrast(image, random.uniform(0.7, 1.3))
            image = TF.adjust_saturation(image, random.uniform(0.7, 1.3))
            image = TF.adjust_hue(image, random.uniform(-0.05, 0.05))

        # Gaussian Blur
        if self.aug.get("gaussian_blur", True) and random.random() < 0.3:
            ks = random.choice([3, 5])
            sigma = random.uniform(0.1, 2.0)
            image = TF.gaussian_blur(image, kernel_size=ks, sigma=sigma)

        # Final resize + DINOv2 normalization
        image = TF.resize(image, list(self.image_size))
        image = TF.normalize(image,
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

        # Random window mask — applied after normalization
        if self.aug.get("random_window_mask", False) and random.random() < 0.5:
            image = self._apply_window_mask(image)

        # Random vertical shift — applied after normalization
        if self.aug.get("random_vertical_shift", False) and random.random() < 0.5:
            image = self._apply_vertical_shift(image)

        # Random horizontal shift — applied after normalization
        if self.aug.get("random_horizontal_shift", False) and random.random() < 0.5:
            image = self._apply_horizontal_shift(image)

        return image

    def _apply_window_mask(self, image):
        """
        Show only a random rectangular window, rest set to normalized black.
        Forces memory to see chain patches at varied positions and contexts.
        Window width: 40-80% of image width.
        Window height: 50-100% of image height.
        """
        C, H, W = image.shape
        black = torch.tensor(NORM_BLACK, dtype=image.dtype).view(3, 1, 1).expand(C, H, W).clone()

        win_w = random.randint(int(W * 0.4), int(W * 0.8))
        win_h = random.randint(int(H * 0.5), H)

        x = random.randint(0, W - win_w)
        y = random.randint(0, H - win_h)

        result = black.clone()
        result[:, y:y + win_h, x:x + win_w] = image[:, y:y + win_h, x:x + win_w]

        return result



    def _apply_vertical_shift(self, image):
        """
        Randomly shift the image up or down, filling the gap with normalized black.
        Forces memory to see the chain at different vertical positions so all
        patch rows learn consistent background features regardless of chain height.
        Shift range: up to vertical_shift_frac (default 30%) of image height in either direction.
        """
        C, H, W = image.shape
        black = torch.tensor(NORM_BLACK, dtype=image.dtype).view(3, 1, 1).expand(C, H, W).clone()

        max_shift = int(H * self.aug.get("vertical_shift_frac", 0.3))
        shift = random.randint(-max_shift, max_shift)

        if shift == 0:
            return image

        result = black.clone()
        if shift > 0:
            result[:, shift:, :] = image[:, :H - shift, :]
        else:
            shift = abs(shift)
            result[:, :H - shift, :] = image[:, shift:, :]

        return result

    def _apply_horizontal_shift(self, image):
        """
        Randomly shift the image left or right, filling the gap with normalized black.
        Simulates the 20-80px chain position shift seen in production images.
        Forces memory to see connector and link patches at varied horizontal positions.
        Shift range: up to horizontal_shift_frac (default 20%) of image width in either direction.
        """
        C, H, W = image.shape
        black = torch.tensor(NORM_BLACK, dtype=image.dtype).view(3, 1, 1).expand(C, H, W).clone()

        max_shift = int(W * self.aug.get("horizontal_shift_frac", 0.2))
        shift = random.randint(-max_shift, max_shift)

        if shift == 0:
            return image

        result = black.clone()
        if shift > 0:
            result[:, :, shift:] = image[:, :, :W - shift]
        else:
            shift = abs(shift)
            result[:, :, :W - shift] = image[:, :, shift:]

        return result
def ok_dir(config_dict):
    """Folder with the OK pictures (training): <data_dir>/images"""
    return os.path.join(config_dict["data"]["data_dir"], "images")


def ng_dir(config_dict):
    """Folder with the NG pictures (only shown in TensorBoard, never trained on): <data_dir>/NG"""
    return os.path.join(config_dict["data"]["data_dir"], "NG")


class DataHandler(Dataset):
    def __init__(self, config_dict, mode="train"):
        """
        Dataset handler for JADE_v3.
        Reads the OK pictures of <data_dir>/images (mode only changes the augmentation).

        :param config_dict: config yaml loaded as dict
        :param mode: "train" or "test"
        """
        self.config_dict = config_dict
        self.mode = mode
        self.image_size = tuple(config_dict["data"]["image_size"])
        self.extensions = config_dict["data"]["image_extensions"]

        self.image_paths = self._load_image_paths(ok_dir(config_dict))
        print(f"[DataHandler] mode={mode} | images: {len(self.image_paths)}")

        aug_config = config_dict["data"].get("augmentation", {})
        self.augmentation = AnomalyAugmentation(self.image_size, aug_config)

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size, antialias=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_image_paths(self, data_dir):
        image_paths = []
        for ext in self.extensions:
            image_paths += glob.glob(os.path.join(data_dir, ext))
        if len(image_paths) == 0:
            raise ValueError(f"[DataHandler] No images found in: {data_dir}\n"
                             f"Put the OK pictures in this folder (and the NG pictures in the NG folder next to it).")
        random.shuffle(image_paths)
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode == "train":
            image_tensor = self.augmentation(image)
        else:
            image_tensor = self.test_transform(image)

        return image_tensor, image_path

    def on_epoch_end(self):
        random.shuffle(self.image_paths)

    @staticmethod
    def preprocess_image(image, image_size=(518, 518)):      # image_size = (height, width), like data.image_size in the config
        """
        Preprocess a single raw BGR image for inference.
        Input:  BGR numpy image
        Output: normalized tensor (1, 3, H, W)
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image_size
        image = cv2.resize(image, (W, H))                       # cv2 wants (width, height)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(image).unsqueeze(0)


def get_dataloader(config_dict, mode="train"):
    dataset = DataHandler(config_dict, mode=mode)
    dataloader = DataLoader(
        dataset,
        batch_size=config_dict["data"]["batch_size"],
        shuffle=(mode == "train"),
        num_workers=config_dict["data"]["num_workers"],
        pin_memory=True,
        drop_last=(mode == "train")
    )
    return dataloader, dataset
