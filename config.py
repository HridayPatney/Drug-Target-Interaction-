import os
from dataclasses import dataclass

@dataclass
class Config:
    # Data parameters
    data_root: str = "data/Davis"
    drug_max_length: int = 100
    protein_max_length: int = 1000
    image_size: int = 256
    
    # Model parameters
    char_dim: int = 64
    conv: int = 32
    drug_kernel: list = (4, 6, 8)
    protein_kernel: list = (4, 8, 12)
    
    # K-gram parameters
    k_gram_sizes: list = (2, 3, 4)
    
    # Training parameters
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    early_stopping_patience: int = 20
    
    # Device settings
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data_root, exist_ok=True)
