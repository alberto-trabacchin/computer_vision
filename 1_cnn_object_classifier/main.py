import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
import matplotlib.pyplot as plt

def disp_image(data_sample, classes):
    img, label = data_sample
    print(f"Label: {classes[label]}")
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

if __name__ == "__main__":
    train_ds_path = Path("/mnt/c/Users/Alberto/Courses/deep_learning_advanced_computer_vision/Datasets/intel_image_classification/seg_train/seg_train/")
    test_ds_path = Path("/mnt/c/Users/Alberto/Courses/deep_learning_advanced_computer_vision/Datasets/intel_image_classification/seg_test/seg_test/")
    train_ds = ImageFolder(root = train_ds_path, transform = transforms.Compose([
        transforms.Resize((150, 150)), transforms.ToTensor()
    ]))
    test_ds = ImageFolder(root = test_ds_path, transform = transforms.Compose([
        transforms.Resize((150, 150)), transforms.ToTensor()
    ]))
    disp_image(train_ds[0], train_ds.classes)