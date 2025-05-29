import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Import my CNN model
from constants import PATH
from model import CNN

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

# Save only the model's learned parameters
torch.save(model.state_dict(), PATH)

print(f"Model weights saved to {PATH}")
