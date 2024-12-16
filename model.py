"""
    Description: Code for A.I. model implementation and utility functions for it.
    Author: Jimmy L. @ SF State MIC Lab
    Date: Summer 2022
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
# tf.get_logger().setLevel('INFO')

# logging configuration
logging.basicConfig(
    level=logging.INFO,  # level of logging - INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # define logging message format
    datefmt='%m/%d/%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)  # Create the 'logger' object, placeholder for logging module name

# Defining the PyTorch model class
class Model(nn.Module):
    def __init__(
        self, 
        num_classes=4, 
        filters=[32, 64], 
        neurons=[512, 128], 
        dropout=0.5,
        kernel_size=(3, 3), 
        input_shape=(1, 8, 52), 
        pool_size=(2, 2)
    ):
        super(Model, self).__init__()
        logger.info("Initializing Model")

        self.num_classes = num_classes
        self.filters = filters
        self.neurons = neurons
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.pool_size = pool_size

        # Defining convolutional layer 1
        self.conv_layer1 = nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=filters[0],
            kernel_size=kernel_size,
            padding=1 # Padding to maintain spatial dimensions
        )
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=pool_size)

        # Defining convolutional layer 2
        self.conv_layer2 = nn.Conv2d(
            in_channels=filters[0],
            out_channels=filters[1],
            kernel_size=kernel_size,
            padding=1 # Padding to maintain spatial dimensions
        )
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=pool_size)

        # Calculating the output dimensions after convolution and pooling
        with torch.no_grad():
            test_input = torch.zeros(1, *input_shape)
            x = self.relu1(self.bn1(self.conv_layer1(test_input)))
            x = self.pool1(x)
            x = self.relu2(self.bn2(self.conv_layer2(x)))
            x = self.pool2(x)
            self.flattened_size = x.numel()
            logger.info(f"Flattened size: {self.flattened_size}")

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, neurons[0])
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(neurons[0], neurons[1])
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(neurons[1], self.num_classes)

    def forward(self, x):
        # Forward pass
        x = self.pool1(self.relu1(self.bn1(self.conv_layer1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv_layer2(x))))
        x = x.view(-1, self.flattened_size)
        x = self.dropout1(self.relu3(self.fc1(x)))
        x = self.dropout2(self.relu4(self.fc2(x)))
        x = self.fc3(x)
        return x  # No softmax at this point

    '''
    Initializes and returns the Model instance.
    Args: 
        num_classes (int): Number of classes to classify
        filters (list): List of output filters for the CNN layers
        neurons (list): List of neurons for the fully connected layers
        dropout (float): Dropout rate
        kernel_size (tuple): Kernel window size for the CNN
        input_shape (tuple): Input shape for the CNN
        pool_size (tuple): Pooling window size
    '''

def get_model(
    num_classes=4, 
    filters=[32, 64], 
    neurons=[512, 128], 
    dropout=0.5,
    kernel_size=(3, 3), 
    input_shape=(1, 8, 52), 
    pool_size=(2, 2)
):
    model = Model(
        num_classes=num_classes, 
        filters=filters, 
        neurons=neurons, 
        dropout=dropout,
        kernel_size=kernel_size, 
        input_shape=input_shape, 
        pool_size=pool_size
    )
    return model

    '''
    Loads the pretrained model based on previous parameters.
    Args: 
        path (str): Path to the pretrained model weights.
        prev_params (list): List of previous model parameters.
    Returns: The new pretrained model with loaded weights
    '''
    


def get_pretrained(path, prev_params):
    base_model = get_model(
        num_classes=prev_params[0],  # 4
        filters=prev_params[1],      # [32, 64]
        neurons=prev_params[2],      # [512, 128]
        dropout=prev_params[3],      # 0.5
        kernel_size=prev_params[4],
        input_shape=prev_params[5],
        pool_size=prev_params[6]
    )
    logger.info(f"Loading Pretrained Model {path}")
    # Load pretrained weights
    base_model.load_state_dict(torch.load(path))
    logger.info("Pretrained Model Loaded")
    return base_model

def create_finetune(base_model, num_classes=4):
    # Build the new model with the given number of classes
    finetune_model = Model(
        num_classes=num_classes,
        filters=base_model.filters,
        neurons=base_model.neurons,
        dropout=base_model.dropout,
        kernel_size=base_model.kernel_size,
        input_shape=base_model.input_shape,
        pool_size=base_model.pool_size
    )

    # Copy parameters from the base model for all layers except the last one
    finetune_model.load_state_dict(base_model.state_dict(), strict=False)

    # Freeze the base model parameters
    for param in finetune_model.parameters():
        param.requires_grad = False

    # Unfreeze the last fully connected layer
    for param in finetune_model.fc3.parameters():
        param.requires_grad = True

    return finetune_model

def train_model(
    model, 
    train_loader,
    val_loader,
    optimizer,
    criterion,
    save_path=None,
    epochs=500,
    patience=80,
    step_size=50, #for StepLR
    gamma=0.1,  #decay_rate
    #lr=0.,    #0.2
    # decay_rate=0.9,
    device='cuda'
):
    '''Trains model with early stopping and learning rate scheduling.
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        optimizer (Optimizer): Optimizer for training
        criterion (Loss): Loss function
        save_path (str): Path to save the model
        epochs (int): Number of epochs
        patience (int): Number of epochs without improvement
        lr (float): Initial learning rate
        decay_rate (float): Learning rate decay rate
        device (str, optional): Device to run the model on (cpu or cuda)'''
    best_loss = float('inf')
    best_accuarcy = 0.0
    patience_counter = 0
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, step_size=step_size, gamma=gamma)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    model.to(device)
    
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        training_accuracies.append(100 * correct / total)
        training_losses.append(train_loss)
        training_accuracies.append(training_accuracies)
        
    '''validation loss'''
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            # Calculate validation accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_loss /= len(val_loader.dataset)
    validation_losses.append(val_loss)
    validation_accuracies.append(100 * correct / total)
    validation_losses.append(val_loss)
    validation_accuracies.append(validation_accuracies)
    logger.info(f"Epoch {epoch+1}/{epochs} - Running Loss: {train_loss:.4f} - Train Acc: {training_accuracies:.4f} - Validation Loss: {val_loss:.4f} - Validation Acc: {validation_accuracies:.4f}")
    
    '''check for improvement'''
    #early stopping, saving best model
    if val_loss < best_loss and validation_accuracies > best_accuarcy:
        best_loss = val_loss
        best_accuarcy = validation_accuracies
        patience_counter = 0
        if save_path:
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")
            
    else:
        patience_counter += 1
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            #return model
            
            
        '''step scheduling'''
        scheduler.step()  #scheduler.step(val_loss)
    return model, training_losses, validation_losses, training_accuracies, validation_accuracies

def plot_logs(training_losses, validation_losses, training_accuracies, validation_accuracies, acc=False, save_path=None):
    '''Plots the training and validation logs.
    Args:
        training_losses (list): Training losses
        validation_losses (list): Validation losses
        acc (bool, optional): Whether to plot accuracy. Defaults to False
        save_path (str, optional): Path to save the plot'''
    plt.figure(figsize=(10, 6))
    if acc:
        plt.title('Model Accuracy')
        plt.plot(training_losses, label='Training Accuracy')
        plt.plot(validation_losses, label='Validation Accuracy')
        plt.ylabel('Accuracy')
    else:
        plt.title('Model Loss')
        plt.plot(training_losses, label='Training Loss')
        plt.plot(validation_losses, label='Validation Loss')
        plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    '''Reltime prediction using pretrained model
    Args:
        model (nn.Module): Model to use for prediction
        sEMG (np.ndarray): sEMG data
        num_channels (int, optional): Number of channels. Defaults to 8
        window_length (int, optional): Window length. Defaults to 32 
        52 is used for now to allow enough spatial dimensions for the CNN
        device (str, optional): Device to run the model on. Defaults to 'gpu'
        Returns:
        int: Predicted gesture index'''
        
def realtime_pred(model, sEMG, num_channels=8, window_length=52, device='gpu'):
    sEMG = np.array(sEMG).reshape(-1, num_channels, window_length)
    sEMG = torch.from_numpy(sEMG).unsqueeze(1).float()
    sEMG = sEMG.to(device)
    
    model.eval()
    
    with torch.no_grad():
        output = model(sEMG)
        _, preds = torch.max(output, 1)
    return preds.item()

def create_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    '''Creates training and validation data loaders.
    Args:
        X_train (np.ndarray): Training data
        y_train (np.ndarray): Training labels
        X_val (np.ndarray): Validation data
        y_val (np.ndarray): Validation labels
        batch_size (int): Batch size
        Returns:
        train_loader (torch.utils.data.DataLoader):
            Training data loader'''
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).long()

    training_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    validation_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
  