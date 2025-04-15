### TASK 2
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from CNN import CNN  # Assuming CNN.py is in the same directory
from sklearn.metrics import roc_auc_score  # Importing roc_auc_score for AUC calculation
import torch.nn.functional as F # Importing functional module for softmax

class Trainer:
    def __init__(self, model, train_loader, test_loader, val_loader=None, optimizer=None, loss_fn=None, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=0.01)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()


    def train(self, epochs=10):
        """ Train the model on the training set.
        Args:
            epochs (int): Number of epochs to train the model.
        """

        self.model.train() # Set the model to training mode


        for epoch in range(epochs):
            running_loss = 0.0 # Initialize running loss for each epoch

            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.to(self.device) # Move data to the device (GPU or CPU)
                labels = labels.to(self.device) # Move target to the device (GPU or CPU)

                self.optimizer.zero_grad() # Zero the gradients before the backward pass

                outputs = self.model(inputs) # Forward pass through the model

                loss = self.loss_fn(outputs, labels) # Compute the loss

                loss.backward() # Backward pass to compute gradients

                self.optimizer.step() # Update the model parameters

                running_loss += loss.item() # Accumulate the loss

            avg_loss = running_loss / (batch_idx + 1) # Average loss for the epoch

            print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(self.train_loader)}') 
            # Print the average loss for the epoch


    def evaluate(self, loader=None):
        """ Evaluate the model on the test set or validation set.
        Args:
            loader (DataLoader): DataLoader for the test or validation set. If None, uses the test_loader.
        Returns:
            float: Accuracy of the model on the test set.
        """
        self.model.eval() # Set the model to evaluation mode

        loader = loader or self.test_loader # Use the test_loader if no loader is provided

        y_true, y_pred = [], [] # Initialize lists to store true and predicted labels

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device) # Move data to the device (GPU or CPU)

                outputs = self.model(images) # Forward pass through the model

                preds = outputs.argmax(dim=1) # Get the predicted labels

                y_true.extend(labels.cpu().numpy()) # Move labels to CPU and convert to numpy array

                y_pred.extend(preds.cpu().numpy()) # Move predictions to CPU and convert to numpy array

        accuracy = accuracy_score(y_true, y_pred) # Compute accuracy

        return accuracy
    
    def evaluate_auc(self, loader= None, pos_class= None):
        self.model.eval() # Set the model to evaluation mode
        loader = loader or self.test_loader # Use the test_loader if no loader is provided
        pos_class = pos_class or 1 # Default positive class is 1
        y_true, y_scores = [], []

        # Iterate through the data loader
        with torch.no_grad(): 
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = F.softmax(self.model(images), dim=1) # Get the softmax probabilities
                y_true.extend((labels == pos_class).cpu().numpy()) # Convert labels to binary (1 for positive class, 0 for negative class)
                y_scores.extend(outputs[:, pos_class].cpu().numpy()) # Get the scores for the positive class

        auc = roc_auc_score(y_true, y_scores) # Compute AUC score

        return auc

# === Running Task 2 Experiments ===
"""
$ python Training-Testing.py 
CNN(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=1024, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=10, bias=True)
)

Training with Validation Split: 0%
100.0%
100.0%
100.0%
100.0%
Epoch 1/10, Loss: 1.843058825873617
Epoch 2/10, Loss: 0.7006924451350658
Epoch 3/10, Loss: 0.5781377986359444
Epoch 4/10, Loss: 0.5174538178611666
Epoch 5/10, Loss: 0.4754209449328085
Epoch 6/10, Loss: 0.4418931762133834
Epoch 7/10, Loss: 0.41398337977463756
Epoch 8/10, Loss: 0.3898790697299087
Epoch 9/10, Loss: 0.3720219781292654
Epoch 10/10, Loss: 0.3550133986164258
Test Accuracy with 0% val split: 0.8582

Training with Validation Split: 10%
Epoch 1/10, Loss: 2.20172238152174
Epoch 2/10, Loss: 0.8433410273909003
Epoch 3/10, Loss: 0.6164166372604845
Epoch 4/10, Loss: 0.5363222860258903
Epoch 5/10, Loss: 0.48469826490816914
Epoch 6/10, Loss: 0.4481734139977191
Epoch 7/10, Loss: 0.4169324422299297
Epoch 8/10, Loss: 0.39204447418098204
Epoch 9/10, Loss: 0.3735168946474367
Epoch 10/10, Loss: 0.3570586396866783
Test Accuracy with 10% val split: 0.8656

Training with Validation Split: 20%
Epoch 1/10, Loss: 2.164117626508077
Epoch 2/10, Loss: 0.8640597453514735
Epoch 3/10, Loss: 0.6543839203516643
Epoch 4/10, Loss: 0.5852551572720209
Epoch 5/10, Loss: 0.5355785616238912
Epoch 6/10, Loss: 0.5007881287932396
Epoch 7/10, Loss: 0.474975066781044
Epoch 8/10, Loss: 0.4463097376227379
Epoch 9/10, Loss: 0.42608228588104247
Epoch 10/10, Loss: 0.4053118579387665
Test Accuracy with 20% val split: 0.8402

Training with Validation Split: 30%
Epoch 1/10, Loss: 2.2903281844007006
Epoch 2/10, Loss: 1.4476598109284493
Epoch 3/10, Loss: 0.704328636464463
Epoch 4/10, Loss: 0.6003620566479873
Epoch 5/10, Loss: 0.5409784692153902
Epoch 6/10, Loss: 0.5024089674653891
Epoch 7/10, Loss: 0.4675851472436565
Epoch 8/10, Loss: 0.4409834634586375
Epoch 9/10, Loss: 0.41987262625400334
Epoch 10/10, Loss: 0.3998038669654042
Test Accuracy with 30% val split: 0.7280

Training with Validation Split: 40%
Epoch 1/10, Loss: 2.2726757306188716
Epoch 2/10, Loss: 1.1542662258258298
Epoch 3/10, Loss: 0.7294745041358535
Epoch 4/10, Loss: 0.6414141761896666
Epoch 5/10, Loss: 0.5898961994406596
Epoch 6/10, Loss: 0.5551875661363822
Epoch 7/10, Loss: 0.5260721616063194
Epoch 8/10, Loss: 0.501393575598461
Epoch 9/10, Loss: 0.4792937152815543
Epoch 10/10, Loss: 0.45967558455086094
Test Accuracy with 40% val split: 0.7088

=== Summary of Results ===
Val Split: 0% | Test Accuracy: 0.8582
Val Split: 10% | Test Accuracy: 0.8656
Val Split: 20% | Test Accuracy: 0.8402
Val Split: 30% | Test Accuracy: 0.7280
Val Split: 40% | Test Accuracy: 0.7088
"""
