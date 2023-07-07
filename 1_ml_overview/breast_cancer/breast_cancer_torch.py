import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(30, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.normalize(x)
        output = self.linear_sigmoid_stack(x)
        return output


if __name__ == "__main__":
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, 
                                                        test_size = 0.2, 
                                                        shuffle = True)
    x_train = torch.FloatTensor(x_train)
    x_test = torch.FloatTensor(x_test)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)
    training_loader = dataloader.DataLoader(x_)

    model = Net()
    y_hat = model(x_train)
    loss_fn = nn.BCELoss()
    loss = loss_fn(y_hat, y_train)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
    