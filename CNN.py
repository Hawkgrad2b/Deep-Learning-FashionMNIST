### TASK 1

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define the layers of the CNN
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1) 
        # Padding is set to 1 to maintain the same spatial dimensions after convolution
        # Kernel size is 3x3, stride is 1 to move one pixel at a time
        # The first layer takes a single-channel input (grayscale image) and outputs 32 channels
        # The output size after this layer will be (32, 28, 28) for a 28x28 input image
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Max pooling with a 2x2 kernel and stride of 2 reduces the spatial dimensions by half
        # The output size after this layer will be (32, 14, 14)
        

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2)
        # The second layer takes 32 channels as input and outputs 64 channels
        # Padding is set to 2 to maintain the same spatial dimensions after convolution
        # The kernel size is 3x3, stride is 1 to move one pixel at a time
        # The output size after this layer will be (64, 14, 14)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Max pooling with a 2x2 kernel and stride of 2 reduces the spatial dimensions by half
        # The output size after this layer will be (64, 7, 7)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # The third layer takes 64 channels as input and outputs 64 channels
        # Padding is set to 1 to maintain the same spatial dimensions after convolution
        # The kernel size is 3x3, stride is 1 to move one pixel at a time
        # The output size after this layer will be (64, 7, 7)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Max pooling with a 2x2 kernel and stride of 2 reduces the spatial dimensions by half
        # The output size after this layer will be (64, 3, 3)

        self.flatten = nn.Flatten()
        # Flatten the output from the convolutional layers to feed into the fully connected layers
        # The output size after flattening will be (64 * 3 * 3)

        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        # The first fully connected layer takes the flattened output and outputs 256 features
        # The input size is 64 * 4 * 4 = 1024 (after flattening)
        self.fc2 = nn.Linear(256, 128)
        # The second fully connected layer takes 256 features as input and outputs 128 features
        # The input size is 256
        self.fc3 = nn.Linear(128, 10)
        # The third fully connected layer takes 128 features as input and outputs 10 features (for 10 classes)
        # The input size is 128


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x))) # Apply the first convolutional layer and max pooling
        x = self.pool2(F.relu(self.conv2(x))) # Apply the second convolutional layer and max pooling
        x = self.pool3(F.relu(self.conv3(x))) # Apply the third convolutional layer and max pooling
        x = self.flatten(x)                   # Flatten the output
        x = F.relu(self.fc1(x))               # Apply the first fully connected layer
        x = F.relu(self.fc2(x))               # Apply the second fully connected layer
        x = self.fc3(x)                       # No softmax here because CrossEntropyLoss expects raw logits
        return x                              # The output is the raw logits for each class