import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
from typing import Dict, Tuple, Optional

class TimeSeriesDataset(Dataset):
    """Dataset for time series data with sliding windows"""
    def __init__(self, data: np.ndarray, window_size: int):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data) - self.window_size
        
    def __getitem__(self, index):
        x = self.data[index:index + self.window_size]
        y = self.data[index + self.window_size]
        return x, y

class ModelTrainer:
    """Trainer for the supply chain attack detection model"""
    def __init__(
        self,
        model: nn.Module,
        window_size: int,
        n_features: int,
        n_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.window_size = window_size
        self.n_features = n_features
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        
        self.losses = {
            "train": [],
            "val": []
        }
        
    def create_data_loaders(
        self, 
        train_data: np.ndarray,
        val_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders"""
        
        # Split into train/val
        val_size = int(len(train_data) * val_split)
        train_data, val_data = train_data[:-val_size], train_data[-val_size:]
        
        # Create datasets
        train_dataset = TimeSeriesDataset(train_data, self.window_size)
        val_dataset = TimeSeriesDataset(val_data, self.window_size)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                loss = self.criterion(output, y)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def fit(
        self, 
        train_data: np.ndarray,
        val_split: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, list]:
        """Train the model"""
        
        train_loader, val_loader = self.create_data_loaders(train_data, val_split)
        
        print(f"Training model for {self.n_epochs} epochs...")
        train_start = time.time()
        
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            
            # Train and validate
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # Store losses
            self.losses["train"].append(train_loss)
            self.losses["val"].append(val_loss)
            
            epoch_time = time.time() - epoch_start
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}")
                print(f"Time: {epoch_time:.2f}s")
                print()
                
        train_time = time.time() - train_start
        print(f"Training completed in {train_time:.2f}s")
        
        return self.losses
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        self.model.eval()
        
        # Convert to tensor
        x = torch.FloatTensor(x).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(x)
            
        return predictions.cpu().numpy() 