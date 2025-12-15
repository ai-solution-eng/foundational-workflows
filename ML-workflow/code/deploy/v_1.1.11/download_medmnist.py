import numpy as np
import os
import logging
from PIL import Image
import pandas as pd
import medmnist
from medmnist.dataset import INFO
from torchvision import transforms
import torch
import pdb
from medmnist import PathMNIST

# Mapping from flag to dataset class
DATASET_MAP = {"pathmnist": PathMNIST}


def preprocess_image_file(img, img_size, n_channels):

    if isinstance(img, torch.Tensor):
        transform = transforms.Normalize(
            mean=[0.5] * n_channels, std=[0.5] * n_channels
        )
        return transform(img).unsqueeze(0) if img.ndim == 3 else transform(img)
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * n_channels, std=[0.5] * n_channels),
            ]
        )
        return transform(img).unsqueeze(0)  # Add batch dimension


def get_datasets(data_flag, apply_transform=True, download=True):

    assert data_flag in DATASET_MAP, f"Invalid dataset name: {data_flag}"
    logging.info(f"Downloading dataset: {data_flag}")

    info = INFO[data_flag]

    DataClass = getattr(medmnist, info["python_class"])
    if apply_transform:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5] * info["n_channels"], std=[0.5] * info["n_channels"]
                ),
            ]
        )
    else:
        transform = None

    return (
        DataClass(split="train", transform=transform, download=download),
        DataClass(split="val", transform=transform, download=download),
        DataClass(split="test", transform=transform, download=download),
        info,
    )


def download_data(args):
    dest_folder = f"{args.destination_path}/{args.data_flag}/{args.data_split}"

    os.makedirs(dest_folder, exist_ok=True)
    try:
        train_ds, val_ds, test_ds, _ = get_datasets(
            args.data_flag,
            apply_transform=False,
            download=False,
        )
    except Exception as e:
        logging.info(e)
        train_ds, val_ds, test_ds, _ = get_datasets(
            args.data_flag,
            apply_transform=False,
            download=True,
        )
    map_split_to_ds = {"train": train_ds, "val": val_ds, "test": test_ds}
    ds = map_split_to_ds[args.data_split]
    img_2_labels = {}
    for j in range(args.num_samples):
        image, label = ds[j]

        image = np.array(image)
        if image.ndim == 3 and image.shape[0] == 3:  # grayscale to RGB
            image = np.transpose(image, (1, 2, 0))  # CHW to HWC

        image = Image.fromarray(image, mode="RGB")

        image.save(f"{dest_folder}/{j:06d}.png")

        img_2_labels[f"{j:06d}"] = label
    df = pd.DataFrame.from_dict(img_2_labels, orient="index", columns=["label"])
    df.index.name = "sample"

    df.to_csv(
        f"{args.destination_path}/{args.data_flag}/{args.data_split}/labels.csv",
        index=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_flag", type=str, default="pathmnist")
    parser.add_argument("--data_split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--destination_path", type=str, default="datasets/medmnist")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    download_data(args)
