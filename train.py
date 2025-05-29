import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a transform to convert images to tensors
transform = transforms.ToTensor()

# Load the training dataset
train_dataset = datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)

# Load the test dataset
test_dataset = datasets.MNIST(
    root="data", train=False, download=True, transform=transform
)

# Create DataLoaders for batching
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Input layer (28x28 pixels → 784)
        self.fc2 = nn.Linear(128, 64)  # Hidden layer
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)  # Output layer (10 classes)
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # No activation here (logits for classification)
        return x


model = NeuralNet().to(device)  # Move model to the appropriate device

criterion = nn.CrossEntropyLoss()  # Suitable for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):  # Number of training cycles
    for images, labels in train_loader:

        images = images.view(images.shape[0], -1).to(device)  # Flatten images
        labels = labels.to(device)  # Move labels to the appropriate device

        optimizer.zero_grad()  # Reset gradients
        output = model(images)  # Forward pass
        loss = criterion(output, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

correct = 0
total = 0
with torch.no_grad():  # Disable gradient tracking for evaluation
    for images, labels in test_loader:

        images = images.view(images.shape[0], -1).to(device)  # Flatten images
        labels = labels.to(device)  # Move labels to the appropriate device

        output = model(images)
        _, predicted = torch.max(output, 1)  # Get highest probability class
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy: {correct / total * 100:.2f}%")
