import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 61, 120)  # Ajuste de dimensiones
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 61 * 61)  # Ajuste de dimensiones
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def simple_training(model, device, criterion, optimizer, train_loader, epochs=10):
    model.to(device)
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("-" * 10)

        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm.tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        print("Loss: {:.4f}".format(epoch_loss))