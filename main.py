from data_loader import FashionMNISTLoader
from model import CNN
from trainer import Trainer
from evaluator import Evaluator
from config import config

import torch.nn as nn
import torch.optim as optim

loader = FashionMNISTLoader(batch_size=config['batch_size'], val_split=0.2)
print(f"Data loader initialized = {loader}")
print("Loading data...")

train_loader, val_loader, test_loader = loader.get_loaders()
print(f"Train loader initialized = {train_loader}")
print(f"Validation loader initialized = {val_loader}")
print(f"Test loader initialized = {test_loader}")
print("Data loaded.")

# Initialize model, loss function, and optimizer
model = CNN()
print(f"CNN model initialized = {model}")

loss_fn = nn.CrossEntropyLoss()
print(f"Loss function initialized = {loss_fn}")

optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
print(f"Optimizer initialized = {optimizer}")

# Train the model
trainer = Trainer(model, optimizer, loss_fn, device=config['device'])
print(f"Trainer initialized = {trainer}")

print("Starting training...")
trainer.train(train_loader, epochs=config['epochs'])
print("Training completed.")

# Evaluate the model
evaluator = Evaluator(model, device=config['device'])
print(f"Evaluator initialized = {evaluator}")

accuracy = evaluator.evaluate_accuracy(test_loader)
print(f"Evaluation completed.")
print(f"Accuracy on test set: {accuracy:.4f}")