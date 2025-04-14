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


# Task 3