import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import multiprocessing

# This import assumes you have the FashionMNISTDataLoader class in 'DataLoader_and_Visualizing.py'
from DataLoader_and_Visualizing import train_loader

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Call this at the beginning

    # Initialize your DataLoader (using num_workers > 0 will trigger multiprocessing)
    def imshow(img):
        img = img / 2 + 0.5  # Unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # Get a batch of training data
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # Display the images in a grid along with their labels
    imshow(torchvision.utils.make_grid(images[:16]))