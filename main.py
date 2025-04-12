from data_loader import FashionMNISTLoader
from model import CNN
from trainer import Trainer
from evaluator import Evaluator
from config import config

import torch
import torch.nn as nn
import torch.optim as optim

loader = FashionMNISTLoader(batch_size=config['batch_size'], val_split=0.2)
train_loader, val_loader, test_loader = loader.get_loaders()

# Initialize model, loss function, and optimizer
model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])

# Train the model
trainer = Trainer(model, optimizer, loss_fn, device=config['device'])
trainer.train(train_loader, epochs=config['epochs'])

# Evaluate the model
evaluator = Evaluator(model, device=config['device'])
accuracy = evaluator.evaluate_accuracy(test_loader)
print(f"Test Accuracy: {accuracy:.4f}")

