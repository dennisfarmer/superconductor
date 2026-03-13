#!/usr/bin/env python

import shutil
from pathlib import Path
from dataloader import get_dataloader
import torch

import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np

from model import PalmModel

# CUDA: Nvidia CUDA-enabled GPU
if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")

# MPS: Apple Silicon
elif torch.backends.mps.is_available():
    print("Using MPS")
    device = torch.device("mps")

# CPU: 
else:
    print("Using CPU")
    device = torch.device("cpu")

def train_model(model_name="palm_up_down", visualize=False):
    dataset_name = model_name
    batch_size = 32

    train_loader = get_dataloader(
        dataset_name=dataset_name,
        partition="train",
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = get_dataloader(
        dataset_name=dataset_name,
        partition="val",
        batch_size=batch_size,
        shuffle=False
    )


    # Get number of classes from dataset
    #num_classes = len(train_loader.dataset.classes) if hasattr(train_loader.dataset, 'classes') else 2
    num_classes = train_loader.dataset.metadata["label"].nunique()
        
    model = PalmModel(input_features=63*2, num_classes=num_classes)
    model.to(device)

    # CrossEntropyLoss combines softmax and negative log-likelihood
    criterion = nn.CrossEntropyLoss()

    #optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Model:\n{model}")

    num_epochs = 50
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        
        for landmarks, labels in train_loader:
            landmarks = landmarks.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()              # reset gradients
            outputs = model(landmarks)         # make a prediction using the model (forward pass)
            loss = criterion(outputs, labels)  # compare predictions to ground truth labels
            loss.backward()                    # calculate gradients (backward pass)
            optimizer.step()                   # update parameters
            
            # Track metrics
            epoch_train_loss += loss.item() * landmarks.size(0)
            _, predicted = torch.max(outputs, 1)
            epoch_train_correct += (predicted == labels).sum().item()
            epoch_train_total += labels.size(0)
        
        # Average training metrics for epoch
        avg_train_loss = epoch_train_loss / epoch_train_total
        avg_train_acc = epoch_train_correct / epoch_train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        
        # set model to eval mode (no updating of weights)
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_correct = 0
        epoch_val_total = 0
        
        with torch.no_grad():
            # check performance on validation dataset
            for landmarks, labels in val_loader:
                landmarks = landmarks.to(device)
                labels = labels.to(device)
                
                outputs = model(landmarks)
                
                loss = criterion(outputs, labels)
                
                epoch_val_loss += loss.item() * landmarks.size(0)
                _, predicted = torch.max(outputs, 1)
                epoch_val_correct += (predicted == labels).sum().item()
                epoch_val_total += labels.size(0)
        
        # Average validation metrics for epoch
        avg_val_loss = epoch_val_loss / epoch_val_total
        avg_val_acc = epoch_val_correct / epoch_val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc*100:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc*100:.2f}%")

    save_path = f"./{model_name}_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"saved model to {save_path}: {val_accuracies[-1]:.2f}% accuracy")

    if visualize:
        # Plot training and validation loss
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        epochs_range = np.arange(len(train_losses))
        plt.plot(epochs_range, train_losses, label='Training Loss', marker='o')
        plt.plot(epochs_range, val_losses, label='Validation Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, np.array(train_accuracies) * 100, label='Training Accuracy', marker='o')
        plt.plot(epochs_range, np.array(val_accuracies) * 100, label='Validation Accuracy', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    print(f"\nFinal Results ({model_name}):")
    print(f"Training Accuracy: {train_accuracies[-1]*100:.2f}%")
    print(f"Validation Accuracy: {val_accuracies[-1]*100:.2f}%")

if __name__ == "__main__":
    #train_model(model_name="palm_up_down")
    #train_model(model_name="palm_hold_release")
    train_model(model_name = "palm_hold_release")
