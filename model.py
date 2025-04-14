import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        # Initialize the layers of the neural network
        super(CNN, self).__init__()

        # Define the sequential layers of the convolutional neural network
        self.net = nn.Sequential(
            # Convolutional Layer 1:
            # - nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1):
            #   - 1: Number of input channels (for grayscale images).
            #   - 32: Number of output channels (number of filters learned by this layer).
            #   - kernel_size=3: The size of the convolutional filter (3x3).
            #   - stride=1: The step size the filter moves across the input.
            #   - padding=1: Adds a 1-pixel border of zeros around the input, helping to preserve spatial dimensions.
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),

            # ReLU Activation 1:
            # - nn.ReLU(): Applies the Rectified Linear Unit activation function element-wise (max(0, x)).
            #   This introduces non-linearity to the network.
            nn.ReLU(),

            # Max Pooling Layer 1:
            # - nn.MaxPool2d(2, 2): Performs max pooling over a 2x2 window with a stride of 2.
            #   This reduces the spatial dimensions (height and width) of the feature maps and provides some translation invariance.
            nn.MaxPool2d(2, 2),

            # Convolutional Layer 2:
            # - nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2):
            #   - 32: Number of input channels (output from the previous layer).
            #   - 64: Number of output channels (number of filters).
            #   - kernel_size=3: Filter size (3x3).
            #   - stride=1: Stride of the filter.
            #   - padding=2: Adds a 2-pixel border of zeros.
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),

            # ReLU Activation 2:
            nn.ReLU(),

            # Max Pooling Layer 2:
            nn.MaxPool2d(2, 2),

            # Convolutional Layer 3:
            # - nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1):
            #   - 64: Number of input channels.
            #   - 64: Number of output channels.
            #   - kernel_size=3: Filter size (3x3).
            #   - stride=1: Stride of the filter.
            #   - padding=1: Adds a 1-pixel border of zeros.
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),

            # ReLU Activation 3:
            nn.ReLU(),

            # Max Pooling Layer 3:
            nn.MaxPool2d(2, 2),

            # Flatten Layer:
            # - nn.Flatten(): Flattens the multi-dimensional output from the convolutional layers into a single one-dimensional tensor.
            #   This is necessary to connect the convolutional part to the fully connected part of the network.
            nn.Flatten(),

            # Fully Connected Layer 1:
            # - nn.Linear(64 * 4 * 4, 256):
            #   - 64 * 4 * 4: The number of input features (64 channels * 4 height * 4 width after the pooling layers).
            #   - 256: The number of output features (number of neurons in this layer).
            nn.Linear(64 * 4 * 4, 256),

            # ReLU Activation 4:
            nn.ReLU(),

            # Fully Connected Layer 2:
            # - nn.Linear(256, 128):
            #   - 256: Number of input features (output from the previous layer).
            #   - 128: Number of output features.
            nn.Linear(256, 128),

            # ReLU Activation 5:
            nn.ReLU(),

            # Fully Connected Layer 3 (Output Layer):
            # - nn.Linear(128, 10):
            #   - 128: Number of input features.
            #   - 10: Number of output features, typically corresponding to the number of classes in a classification problem.
            nn.Linear(128, 10)
            
            # Note: A Softmax activation is typically applied to the output of the last linear layer
            # during training (implicitly by the CrossEntropyLoss) or during inference to get probability distributions.
            # It's not explicitly included as a separate layer here.
        )

    def forward(self, x):
        # Define the forward pass of the network. This specifies how the input data flows through the layers.
        return self.net(x)

# The 'CNN' class now defines a convolutional neural network architecture.
# When an input tensor 'x' is passed to an instance of this class (e.g., 'model(x)'),
# it will go through each layer defined in 'self.net' in sequential order.