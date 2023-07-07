import numpy as np
import torch
from torch.utils.data import random_split
from typing import Tuple
from sklearn.datasets import load_breast_cancer

class CancerDataset(torch.utils.data.Dataset):
    def __init__(self):
        dataset = load_breast_cancer()
        self.data = dataset["data"]
        self.labels = dataset["target"]
        self.size = len(dataset["data"])
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        return (
            self.data[index],
            self.labels[index]
        )

def make_loader(ds, batch_size = 4, num_workers = 4):
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True
    )
    return loader

if __name__ == "__main__":
    ds = CancerDataset()
    ds_train, ds_test, ds_val = random_split(ds, [0.3, 0.3, 0.4])
    train_loader = make_loader(ds_train)
    test_loader = make_loader(ds_test)
    val_loader = make_loader(ds_val)

    print(f"Dataset length: {len(ds)}.")
    print(f"Dataset's first element:\n {ds[0]}\n")
    print(f"Train dataset length: {len(ds_train)}.")
    print(f"Train dataset's first element:\n {ds_train[0]}\n")
    print(f"Test dataset length: {len(ds_test)}.")
    print(f"Test dataset's first element:\n {ds_test[0]}\n")
    print(f"Validation dataset length: {len(ds_val)}.")
    print(f"Validation dataset's first element:\n {ds_val[0]}\n")

    for batch in iter(train_loader):
        data, labels = batch
    
    for i, batch in enumerate(train_loader):
        data, labels = batch

    print("Last train batch:")
    print(data)