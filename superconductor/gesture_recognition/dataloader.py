
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
from typing import Union
from pathlib import Path
import pandas as pd
import torch

class ImageDataset(Dataset):
    def __init__(
            self,
            directory: Union[str, Path],
            partition: str = "train",
            indices: list[int] = None,
        ):
            self.partition = partition
            if partition not in ("train", "test", "val"):
                raise ValueError(f"Invalid partition specified - {partition}")
            self.directory = Path(directory)
            self.img_directory = self.directory / partition
            metadata = pd.read_csv(self.directory / "gestures.csv")
            #self.num_classes = metadata["label"].nunique()
            
            if indices is not None:
                # Use provided indices (for stratified split)
                self.metadata = metadata.iloc[indices].reset_index(drop=True)
            else:
                # Use partition column
                self.metadata = metadata[metadata["partition"] == partition]
                self.metadata.reset_index(drop=True, inplace=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        row = self.metadata.iloc[index]
        label = row["label"]
        tensor_idx = row["index"]
        tensor_path = str(self.img_directory / f"{tensor_idx}.pt")
        
        landmark_tensor = torch.load(tensor_path)
        
        if landmark_tensor is None:
            raise RuntimeError(f"Failed to load image at {tensor_path}. File may be missing or corrupted.")

        return landmark_tensor, label


def get_dataloader(
    dataset_name: str = "custom_dataset",
    partition: str = "train",
    batch_size: int = 1,
    num_workers = 0,
    shuffle: bool = None,
    train_val_split: float = 0.8,
    random_state: int = 42,
) -> DataLoader:
    """
    dataset: "custom_dataset"|"palm_up_down"|"palm_hold_release"
    partition: "train" | "val"

    Performs stratified train/val split based on labels.
    """

    assert (partition == "train") or (partition == "val")

    directory = f"{dataset_name}_processed"
    metadata = pd.read_csv(Path(directory) / "gestures.csv")
    train_val_metadata = metadata[metadata["partition"] != "test"]
    
    if len(train_val_metadata) > 0:
        labels = train_val_metadata["label"].values
        indices = train_val_metadata.index.values
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1-train_val_split, random_state=random_state)
        train_idx, val_idx = next(sss.split(indices, labels))
        
        if partition == "train":
            selected_indices = indices[train_idx]
        else:
            selected_indices = indices[val_idx]
    else:
        selected_indices = train_val_metadata.index.values
    
    dataset = ImageDataset(
        directory=directory,
        partition="train",
        indices=list(selected_indices)
    )

    loader = DataLoader(
        dataset = dataset,
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=shuffle
        )
        
    return loader