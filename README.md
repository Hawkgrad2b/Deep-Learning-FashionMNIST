# Task 1: Implementing CNN from scratch [30 Points]
### Please implement the following CNN architecture manually by using PyTorch:

Layer Type Configuration
Input 28×28 Gray-Scale Image
Conv1 32 filters, 3×3, stride 1, padding 1
MaxPool 2×2, stride 2
Conv2 64 filters, 3×3, stride 1, padding 2
MaxPool 2×2, stride 2
Conv3 64 filters, 3×3, stride 1, padding 1
MaxPool 2×2, stride 2
Flatten —
FC1 Fully Connected (Output Size:
256)
FC2 Fully Connected (Output Size:
128)
FC3 Fully Connected (Output Size: 10)
Softmax —
In your report, please print and show the configuration of your implemented CNN for
checking whether this model has been correctly implemented.