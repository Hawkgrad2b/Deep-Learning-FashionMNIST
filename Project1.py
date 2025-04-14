from CNN import CNN
from FashionMNISTLoader import FashionMNISTLoader
from TrainValTestSplit import Trainer
import torch
import logging

# Configure logging
logging.basicConfig(
    filename='experiment.log',  # Log to this file
    level=logging.INFO,        # Log messages with INFO level or higher
    format='%(asctime)s - %(levelname)s - %(message)s'
)

'''

# Task 1
logging.info("=== Running Task 1 Experiments ===")
model = CNN()
logging.info(f"Model Architecture: {model}") # Log the model architecture
logging.info("Model successfully initialized.") # Log successful initialization


# Task 2
logging.info("=== Running Task 2 Experiments ===")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Set device to GPU if available, else CPU
splits = [0.0, 0.1, 0.2, 0.3, 0.4] # Validation splits to test
results = []

for split in splits:
    logging.info(f"\nTraining with Validation Split: {int(split*100)}%")
    
    loader = FashionMNISTLoader(batch_size=64, val_split=split) # Initialize the data loader with the current split

    train_loader, val_loader, test_loader = loader.get_loaders() # Get the data loaders
    logging.info(f"DataLoaders initialized with batch size 64 and val_split {split}")

    model = CNN() # Initialize the model

    trainer = Trainer(model, train_loader, test_loader, val_loader, device=device) # Initialize the trainer

    trainer.train(epochs=10) # Train the model
    logging.info(f"Training completed for {int(split*100)}% validation split.")

    acc = trainer.evaluate() # Evaluate the model on the test set

    logging.info(f"Test Accuracy with {int(split*100)}% val split: {acc:.4f}")
    results.append((split, acc))


logging.info("\n=== Summary of Results ===")
print("\n=== Summary of Results ===")
for split, acc in results:
    logging.info(f"Val Split: {int(split*100)}% | Test Accuracy: {acc:.4f}")
    print(f"Val Split: {int(split*100)}% | Test Accuracy: {acc:.4f}")

'''

# Task 3
logging.info("\n=== Running Task 3 Experiments ===")\

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Set device to GPU if available, else CPU

best_val_split = 0.1 # Assuming the best validation split is 0.1 based on previous results
learning_rates = [0.001, 0.01, 0.1, 1, 10] # Learning rates to test
lr_results = []

for lr in learning_rates:
    logging.info(f"\nTraining with Learning Rate: {lr}")
    
    loader = FashionMNISTLoader(batch_size=64, val_split=best_val_split) # Initialize the data loader with the best split

    train_loader, val_loader, test_loader = loader.get_loaders() # Get the data loaders
    logging.info(f"DataLoaders initialized with batch size 64 and val_split {best_val_split}")


    model = CNN() # Initialize the model
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) # Initialize the optimizer with the current learning rate
    trainer = Trainer(model, train_loader, test_loader, val_loader, optimizer, device=device) # Initialize the trainer

    trainer.train(epochs=10) # Train the model
    logging.info(f"Training completed for learning rate {lr}.")

    acc = trainer.evaluate() # Evaluate the model on the test set

    logging.info(f"Test Accuracy with learning rate {lr}: {acc:.4f}")
    lr_results.append((lr, acc))

print("\n=== Summary of Learning Rate Results ===")
for lr, acc in lr_results:
    print(f"Learning Rate: {lr} | Test Accuracy: {acc:.4f}")