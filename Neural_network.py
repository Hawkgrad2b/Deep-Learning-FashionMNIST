import torch.nn as nn 
import torch.optim as optim
import torch

input_size = 2 
hidden_size = 8
output_size = 1

# define the nural network: it only has one hidden layer and output layer
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # input layer to hidden layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # hidden layer to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # apply ReLU activation function
        x = self.fc2(x)  # output layer
        return x
    
net = SimpleNet()  # create an instance of the neural network

# Loss function: How good the model is doing in forward pass
# Optimizer: How to update the model parameters in backward pass

# When we start backward passes, we first zero the gradients in the optimer 
# by optimer.zero_grad() before the backward pass. MUST DO THIS = del prev grad

# calculate the gradient with loss.backward()
# update the parameters with optimizer.step()

print(net)

# define the loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss function

# Stochastic Gradient Descent optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)  

# prepare dummy data and labels
data = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float32)
labels = torch.tensor([[0.], [1.]], dtype=torch.float32)

# training loop

for epoch in range(500):
    # forward pass
    outputs = net(data)
    loss = criterion(outputs, labels)  # calculate the loss

    # IMPORTANT: zero the gradients before the backward pass
    optimizer.zero_grad()  # clear previous gradients

    # backward pass
    loss.backward()  # compute gradients
    optimizer.step()  # update parameters

    # print the loss for this epoch
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/500], Loss: {loss.item():.4f}')
    
