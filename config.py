import torch

config = {
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}