import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(64)

        self.conv7 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 5 * 5, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # Apply first convolution and ReLU
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        x = self.relu(self.bn4(self.conv4(x)))  # Apply second convolution and ReLU
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))

        x = self.relu(self.bn7(self.conv7(x)))  # Apply third convolution and ReLU
        
        x = x.view(x.size(0), -1)
        x = torch.log_softmax(self.fc1(x), dim=1)

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

PATH = "weights.pth" # .pth or .pt are common extensions

# Save only the model's learned parameters
torch.save(model.state_dict(), PATH)

print(f"Model weights saved to {PATH}")
