import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNetBig(nn.Module):
    """
    ConvNet Model for all features data. Variable number of channels
    """

    def __init__(self, log_softmax=False, softmax=True, dropout=0., channels=3):
        super(ConvNetBig, self).__init__()
        self.log = log_softmax
        self.soft = softmax
        # Input: 3 channels, 8 output channels, 3x3 conv -> 8x3x3 output
        self.conv1 = nn.Conv2d(channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        # Predict all three frequencies, use Softmax?
        self.fc4 = nn.Linear(32, 3)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.dropout_conv = nn.Dropout2d(p=0.1)
        self.dropout_fcn = nn.Dropout(p=dropout)
        self.gradients = None  # Placeholder for gradients

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout_conv(x)
        x = F.relu(self.conv2(x))
        x = self.dropout_conv(x)
        x = F.relu(self.conv3(x))
        # Hook to capture the gradients
        h = x.register_hook(self.activations_hook)
        x = self.dropout_conv(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.dropout_fcn(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fcn(x)
        x = F.relu(self.fc3(x))
        x = self.dropout_fcn(x)
        x = self.fc4(x)
        if self.log:
            return self.log_softmax(x)
        elif self.soft:
            return self.softmax(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    # Method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # Method for the activation extraction
    def get_activations(self, x):
        activations = []
        x = F.relu(self.conv1(x))
        activations.append(x)
        x = self.dropout_conv(x)
        x = F.relu(self.conv2(x))
        activations.append(x)
        x = self.dropout_conv(x)
        x = F.relu(self.conv3(x))
        activations.append(x)
        return activations
