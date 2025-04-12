import torch
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train(self, dataloader, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")