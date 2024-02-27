import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

from PIL import Image
from tqdm import tqdm

def reindex_df(all_judgments, image_info):
    def get_idx(name):
        return image_info.loc[image_info["Name"].str.lower() == name].index.values.astype(int)[0]

    all_judgments["left_id"] = all_judgments["left"].map(lambda x: get_idx(x))
    all_judgments["right_id"] = all_judgments["right"].map(lambda x: get_idx(x))
    #save indexed_judgements
    judgements_index = all_judgments[["left_id", "right_id", "choice"]]
    judgements_index.head()
    judgements_index.to_csv("data/all_judgments_indexed.csv", index=False)

def rename_images(image_info):
    for img_name, index in tqdm(zip(image_info["Name"], image_info.index)):
        source_path = "images/" + img_name
        dest_path = "rename_images/image_" + str(index) + ".jpg"
        shutil.copy(source_path, dest_path)

IMAGE_SIZE = (448, 448)
def get_transform(evaluation = False):
    if evaluation: 
        transform = v2.Compose([
            v2.Resize(size = IMAGE_SIZE, antialias=True),
            v2.RandomHorizontalFlip(p = 0.5),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # v2.FiveCrop(IMAGE_SIZE[0]/2)
        ])
    else:
        transform = v2.Compose([
            v2.Resize(size = IMAGE_SIZE, antialias=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # v2.FiveCrop(IMAGE_SIZE[0]/2)
        ])
    return transform


class JudgementDataset(Dataset):
    def __init__(self, judgements, transform, five_crop=False, return_index = False):
        self.judgements = judgements
        self.transform = transform
        self.five_crop = five_crop
        self.return_index = return_index

    def __len__(self):
        return self.judgements.shape[0]

    def __getitem__(self, index):
        left_image_id = self.judgements.loc[index, "left_id"]
        right_image_id = self.judgements.loc[index, "right_id"]
        choice = self.judgements.loc[index, "choice"]
        with Image.open("images/image_" + str(left_image_id) + ".jpg") as image:
            image_np_left = np.array(image, dtype=np.float32)
        image_np_left = image_np_left.transpose((2, 0, 1))
        image_np_left = torch.from_numpy(image_np_left)
        if self.transform:
            image_np_left = self.transform(image_np_left)
            if self.five_crop:
                image_np_left_crop = v2.FiveCrop(IMAGE_SIZE[0]/2)(image_np_left)
                image_np_left_crop = torch.stack(image_np_left_crop)

        with Image.open("images/image_" + str(right_image_id) + ".jpg") as image:
            image_np_right = np.array(image, dtype=np.float32)
        image_np_right = image_np_right.transpose((2, 0, 1))
        image_np_right = torch.from_numpy(image_np_right)
        if self.transform:
            image_np_right = self.transform(image_np_right)
            if self.five_crop:
                image_np_right_crop = v2.FiveCrop(IMAGE_SIZE[0]/2)(image_np_right)
                image_np_right_crop = torch.stack(image_np_right_crop)
        cls = torch.from_numpy(np.asarray(choice))

        if self.five_crop:
            if self.return_index:
                return (image_np_left, image_np_left_crop), (image_np_right, image_np_right_crop), cls, index
            else:
                return (image_np_left, image_np_left_crop), (image_np_right, image_np_right_crop), cls
        else:
            if self.return_index:
                return image_np_left, image_np_right, cls, index
            else:
                return image_np_left, image_np_right, cls
        


if __name__ == "__main__":
    # Testing transform
    with Image.open("images/image_" + str(0) + ".jpg") as image:
        image_np = np.array(image, dtype=np.float32)
    image_np = image_np.transpose((2, 0, 1))
    image_np = torch.from_numpy(image_np)
    new = get_transform(image_np)
    print("Shape after transformation:", new.shape)

    # Test Dataloader
    judgements_df = pd.read_csv("data/all_judgements_indexed.csv")
    val_size = 0.2
    train_size = int(judgements_df.shape[0] * (1-val_size))
    transform = get_transform()
    train_judgements = pd.read_csv("data/train_judgements.csv")
    test_judgements = pd.read_csv("data/test_judgements.csv")

    train_set = JudgementDataset(judgements=train_judgements, transform=transform, five_crop=True)
    test_set = JudgementDataset(judgements=test_judgements, transform=transform, five_crop=True)
    val_set = JudgementDataset(judgements=judgements_df.iloc[train_size:], transform=transform, five_crop=True)
    batch_size = 4
    num_workers = 4

    train_loader = DataLoader(
        dataset= train_set,
        batch_size= batch_size,
        num_workers= num_workers,
        shuffle= True,
        pin_memory= True,
    )

    test_loader = DataLoader(
        dataset= test_set,
        batch_size= batch_size,
        num_workers= num_workers,
        shuffle= True,
        pin_memory= True,
    )

    #Testing DataLoader
    for left, right, cls in train_loader:
        print("left image shape:", left[0].shape)
        print("left image crop shape:", left[1].shape)
        print("right image shape:", right[0].shape)
        print("right image shape:", right[1].shape)
        print("cls shape:", cls.shape)
        image_global = left[0][0].numpy()
        image_global = np.array(image_global, dtype = np.int64)
        image_global = image_global.transpose((1,2,0))
        image_patch = left[1][0].numpy()
        image_patch = np.array(image_patch, dtype = np.int64)
        print(image_patch.shape)
        image_patch = image_patch.transpose((0,2,3,1))
        fig, axes = plt.subplots(1, 6, figsize=(30, 30))
        axes[0].imshow(image_global)
        axes[1].imshow(image_patch[0])
        axes[2].imshow(image_patch[1])
        axes[3].imshow(image_patch[2])
        axes[4].imshow(image_patch[3])
        axes[5].imshow(image_patch[4])
        # plt.imshow(image)
        break
