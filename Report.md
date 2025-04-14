### Task 1: CNN Architecture Implementation

The CNN model was implemented in PyTorch as specified:

- **Input:** 28x28 grayscale image
- **Conv1:** 32 filters, 3×3, stride 1, padding 1
- **MaxPool:** 2×2, stride 2
- **Conv2:** 64 filters, 3×3, stride 1, padding 2
- **MaxPool:** 2×2, stride 2
- **Conv3:** 64 filters, 3×3, stride 1, padding 1
- **MaxPool:** 2×2, stride 2
- **Fully Connected Layers:** 256 → 128 → 10
- **Output:** Logits (softmax will be applied during evaluation)

The model summary confirms the layer configuration and output shapes.
