import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, precision_score, recall_score

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def calculate_metrics(y_true, y_pred, y_score):
    """Calculate evaluation metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    # Calculate AUC
    auc_score = roc_auc_score(y_true, y_score)
    
    # Calculate AUPR
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    aupr = auc(recall_curve, precision_curve)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc_score,
        'aupr': aupr
    }

def plot_training_history(train_losses, val_losses, metrics_history, save_dir):
    """Plot training history"""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()
    
    # Plot metrics
    metrics = ['accuracy', 'precision', 'recall', 'auc', 'aupr']
    for metric in metrics:
        plt.figure(figsize=(10, 5))
        plt.plot([history[metric] for history in metrics_history])
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.title(f'Validation {metric.capitalize()}')
        plt.savefig(os.path.join(save_dir, f'{metric}.png'))
        plt.close()

def visualize_attention(model, dataloader, save_dir, num_samples=5):
    """Visualize attention weights"""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            drug = batch['drug'].to(device)
            protein = batch['protein'].to(device)
            image = batch['image'].to(device)
            
            # Forward pass
            _ = model(drug, protein, image)
            
            # Get attention weights (modify this based on your model's attention mechanism)
            try:
                # Extract attention weights from the model
                drug_att = model.drug_attention_layer(model.fused_drug_features.permute(0, 2, 1))
                protein_att = model.protein_attention_layer(model.proteinConv.permute(0, 2, 1))
                
                # Visualize drug attention
                plt.figure(figsize=(10, 5))
                plt.imshow(drug_att[0].cpu().numpy(), aspect='auto', cmap='viridis')
                plt.colorbar()
                plt.title('Drug Attention Weights')
                plt.savefig(os.path.join(save_dir, f'drug_attention_{i}.png'))
                plt.close()
                
                # Visualize protein attention
                plt.figure(figsize=(10, 5))
                plt.imshow(protein_att[0].cpu().numpy(), aspect='auto', cmap='viridis')
                plt.colorbar()
                plt.title('Protein Attention Weights')
                plt.savefig(os.path.join(save_dir, f'protein_attention_{i}.png'))
                plt.close()
            except Exception as e:
                print(f"Error visualizing attention: {e}")

def setup_data_directories(config):
    """Ensure all necessary directories exist"""
    # Create main data directory
    os.makedirs(config.data_root, exist_ok=True)
    
    # Create directory for drug images
    image_dir = os.path.join(config.data_root, f"drug_images_{config.image_size}")
    os.makedirs(image_dir, exist_ok=True)
    
    # Check if Davis.txt exists
    davis_file = os.path.join(config.data_root, "Davis.txt")
    if not os.path.exists(davis_file):
        print(f"WARNING: Davis.txt not found at {davis_file}")
        print("Please ensure the file exists before running the model")
    else:
        print(f"Found Davis.txt at {davis_file}")
    
    return True
