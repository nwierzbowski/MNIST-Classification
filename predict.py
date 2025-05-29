import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from constants import PATH
from model import CNN

transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="data", train=False, download=True, transform=transform
)
# Use a smaller batch size for visualization if you only want to see one at a time,
# or keep it as 1 to see individual images.
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


model = CNN()

print("Loading model weights...")

model.load_state_dict(torch.load(PATH))

model.eval()

print("Model loaded successfully.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model.to(device)  # Move model to the appropriate device

print("\nStarting prediction visualization. Close each image window to see the next.")
print("Press Ctrl+C in the console to stop the visualization at any time.")

correct_predictions = 0
total_images = 0

with torch.no_grad(): # Disable gradient tracking for inference
    for i, (images, labels) in enumerate(test_loader):
        if i >= 1000: # Limit to first 20 images for demonstration, remove or increase for more
            break

        # Move image and label to the same device as the model
        # images = images.view(images.shape[0], -1).to(device)
        images = images.view(images.size(0), 1, 28, 28).to(device)
        labels = labels.to(device)

        # Get prediction
        output = model(images)
        probabilities = torch.softmax(output, dim=1) # Convert logits to probabilities
        _, predicted = torch.max(output, 1) # Get the class with the highest logit

        # Get the actual image data for plotting (move back to CPU and reshape)
        image_to_show = images.squeeze().cpu().numpy()
        true_label = labels.item()
        predicted_label = predicted.item()

        # Display the image and prediction
        plt.figure(figsize=(4, 4))
        plt.imshow(image_to_show, cmap='gray')
        plt.title(f"True: {true_label} | Predicted: {predicted_label}")
        plt.axis('off')
        plt.show() # This will open a new window and pause execution until it's closed

        # Keep track of accuracy for the visualized batch
        if predicted_label == true_label:
            correct_predictions += 1
        total_images += 1

print(f"\nVisualization finished. Total images viewed: {total_images}")
if total_images > 0:
    print(f"Accuracy on viewed images: {correct_predictions / total_images * 100:.2f}%")