import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from model import MultimodalHyperAttentionDTI
from data_processor import get_data_loaders
from utils import EarlyStopping, calculate_metrics, plot_training_history, visualize_attention, setup_data_directories
from config import Config

def train(model, train_loader, val_loader, criterion, optimizer, config):
    """Train the model"""
    device = torch.device(config.device)
    model = model.to(device)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        verbose=True,
        path=os.path.join(config.checkpoint_dir, 'best_model.pt')
    )
    
    # Training history
    train_losses = []
    val_losses = []
    metrics_history = []
    
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        
        # Training
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Train]')
        for batch in train_bar:
            # Get data
            drug = batch['drug'].to(device)
            protein = batch['protein'].to(device)
            image = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(drug, protein, image)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * drug.size(0)
            train_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average training loss
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        val_loss, metrics = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        metrics_history.append(metrics)
        
        # Print statistics
        print(f'Epoch {epoch+1}/{config.epochs}:')
        print(f' Train Loss: {train_loss:.4f}')
        print(f' Val Loss: {val_loss:.4f}')
        print(f' Val Metrics: {metrics}')
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, 'best_model.pt')))
    
    # Plot training history
    plot_training_history(train_losses, val_losses, metrics_history, config.log_dir)
    
    return model, train_losses, val_losses, metrics_history

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    val_loss = 0.0
    y_true = []
    y_pred = []
    y_score = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Get data
            drug = batch['drug'].to(device)
            protein = batch['protein'].to(device)
            image = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(drug, protein, image)
            loss = criterion(outputs, labels)
            
            # Update statistics
            val_loss += loss.item() * drug.size(0)
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            
            # Collect for metrics calculation
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_score.extend(probabilities.cpu().numpy())
    
    # Calculate average validation loss
    val_loss = val_loss / len(dataloader.dataset)
    
    # Calculate metrics
    metrics = calculate_metrics(np.array(y_true), np.array(y_pred), np.array(y_score))
    
    return val_loss, metrics

def test(model, test_loader, criterion, config):
    """Test the model"""
    device = torch.device(config.device)
    model = model.to(device)
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(config.checkpoint_dir, 'best_model.pt')))
    
    # Evaluate on test set
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    
    print("Test Results:")
    print(f" Loss: {test_loss:.4f}")
    print(f" Metrics: {test_metrics}")
    
    # Visualize attention weights
    visualize_attention(model, test_loader, os.path.join(config.log_dir, 'attention'), num_samples=5)
    
    return test_loss, test_metrics

def main():
    # Load configuration
    config = Config()
    
    # Setup directories and check data files
    setup_data_directories(config)
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    # Initialize model
    model = MultimodalHyperAttentionDTI(config)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Train model
    model, train_losses, val_losses, metrics_history = train(
        model, train_loader, val_loader, criterion, optimizer, config
    )
    
    # Test model
    test_loss, test_metrics = test(model, test_loader, criterion, config)
    
    # Save results
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'metrics_history': metrics_history,
        'test_loss': test_loss,
        'test_metrics': test_metrics
    }
    
    # Save results to file
    import json
    with open(os.path.join(config.log_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
