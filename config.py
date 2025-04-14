import torch

config = {
    # Number of samples processed in one training iteration
    "batch_size": 64,

    # Number of complete passes through the training dataset
    "epochs": 10,

    # Step size for the optimizer during weight updates
    "learning_rate": 0.001, 
    
    # Device to use for training (GPU if available, else CPU)
    "device": "cuda" if torch.cuda.is_available() else "cpu" 
}