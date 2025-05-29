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


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 10)  # Output layer (10 classes)
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # Apply first convolution and ReLU
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))  # Fully connected layer with ReLU
        x = self.fc2(x)  # No activation here (logits for classification)
        return x


model = CNN().to(device)  # Move model to the appropriate device

criterion = nn.CrossEntropyLoss()  # Suitable for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):  # Number of training cycles
    for images, labels in train_loader:
        images = images.view(images.size(0), 1, 28, 28).to(device)
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
        images = images.view(images.size(0), 1, 28, 28).to(device)  # Maintain 4D shape for CNN
        labels = labels.to(device)  # Move labels to the appropriate device

        output = model(images)
        _, predicted = torch.max(output, 1)  # Get highest probability class
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy: {correct / total * 100:.2f}%")
