import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import torch
import pandas as pd
from typing import Tuple
from sklearn.datasets import load_breast_cancer
import numpy as np

class CancerDataset(torch.utils.data.Dataset):
    def __init__(self):
        dataset = load_breast_cancer()
        self.data = dataset["data"]
        self.labels = dataset["target"].astype(np.float64).reshape(-1, 1)
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(30, 10),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.normalize(x)
        output = self.linear_sigmoid_stack(x)
        return output
    

def train(train_loader, model, loss_fun, optimizer, device, epochs):
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            data, labels = batch
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            predicts = model(data)
            loss = loss_fun(predicts, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss
        losses.append(running_loss)
        if (epoch % 10 == 0):
            print(f"Loss at epoch {epoch + 1}: {running_loss}")
        running_loss = 0.0
    return losses


if __name__ == "__main__":
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")
    ds = CancerDataset()
    ds_train, ds_test, ds_val = random_split(ds, [0.3, 0.3, 0.4])
    train_loader = make_loader(ds_train)
    test_loader = make_loader(ds_test)
    val_loader = make_loader(ds_val)
    model = Net().double()
    model.to(device)
    loss_fun = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    train(train_loader, model, loss_fun, optimizer, device, epochs = 1000)