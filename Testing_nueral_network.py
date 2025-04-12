import torch

from Neural_network import net
from DataLoader_and_Visualizing import test_loader
import multiprocessing


if __name__ == '__main__':
    multiprocessing.freeze_support()  # Call this at the beginning
    # Test the neural network
    correct = 0
    total = 0

    # Load the saved model and Set the model to evaluation mode
    net.load_state_dict(torch.load("best_fashionmnist_model.pth"))
    net.eval()

    # Disable gradient calculation
    with torch.no_grad():
        from CNN import device
        for inputs, labels in test_loader:

        

            # Move the inputs and labels to the GPU if available
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = net(inputs)

            # Get the predicted class
            _, predicted = torch.max(outputs.data, 1)

            # Update the total number of samples and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate the accuracy
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")