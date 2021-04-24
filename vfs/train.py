import argparse
import torch
from models import *
import numpy as np

def train_feature_extractor(device, dataloader, lr=0.001, num_epochs=10):
    feature_extractor = Conv4()
    feature_extractor.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=lr)

    feature_extractor.train()
    for epoch in range(num_epochs):
        train_loss = 0.
        for batch_idx, (ims, labels) in enumerate(dataloader, 0):
            ims, labels = ims.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = feature_extractor(ims)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Average Loss {round(train_loss / len(dataloader.dataset), 3)}")
    
    return feature_extractor

def test_feature_extractor(feature_extractor, device, dataloader):
    feature_extractor.to(device)

    correct, total = 0, 0

    feature_extractor.eval()
    with torch.no_grad():
        for ims, labels in dataloader:
            ims, labels = ims.to(device), labels.to(device)
            outputs = feature_extractor(ims)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Overall Accuracy {round(100. * (correct / total), 3)}")

def extract_features(feature_extractor, device, dataloader, bottleneck="layers.4"):
    features = []
    with torch.no_grad():
        for ims, _ in dataloader:
            ims = ims.to(device)
            outputs = feature_extractor(ims)
            features.append(feature_extractor.activations[bottleneck].detach().cpu().numpy())
    return np.array(features)

def generator_loss(y_true, y_pred):
    pass

def train_generator(device, dataloader, optimizer):
    pass