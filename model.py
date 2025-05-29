import torch
import torch.nn as nn

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