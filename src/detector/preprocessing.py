import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler

class SupplyChainDataPreprocessor:
    """Preprocessor for supply chain data"""
    def __init__(
        self,
        feature_columns: List[str],
        target_column: str,
        window_size: int = 100,
        scaler: Optional[StandardScaler] = None
    ):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.window_size = window_size
        self.scaler = scaler if scaler else StandardScaler()
        
    def fit_transform(
        self, 
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit scaler and transform data"""
        
        # Extract features and target
        X = data[self.feature_columns].values
        y = data[self.target_column].values
        
        # Fit and transform features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y)
        
        return X_seq, y_seq
    
    def transform(
        self, 
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform data using fitted scaler"""
        if not hasattr(self.scaler, 'mean_'):
            raise ValueError("Scaler has not been fitted. Call fit_transform first.")
            
        # Extract features and target
        X = data[self.feature_columns].values
        y = data[self.target_column].values
        
        # Transform features
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y)
        
        return X_seq, y_seq
    
    def _create_sequences(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series data"""
        
        sequences = []
        targets = []
        
        for i in range(len(X) - self.window_size):
            seq = X[i:i + self.window_size]
            target = y[i + self.window_size]
            sequences.append(seq)
            targets.append(target)
            
        return np.array(sequences), np.array(targets)
    
    def inverse_transform_features(
        self, 
        X_scaled: np.ndarray
    ) -> np.ndarray:
        """Inverse transform scaled features"""
        return self.scaler.inverse_transform(X_scaled)
    
    @staticmethod
    def load_data(
        filepath: str,
        feature_columns: List[str],
        target_column: str
    ) -> pd.DataFrame:
        """Load data from file"""
        try:
            data = pd.read_csv(filepath)
            required_columns = feature_columns + [target_column]
            
            if not all(col in data.columns for col in required_columns):
                missing = [col for col in required_columns if col not in data.columns]
                raise ValueError(f"Missing columns in data: {missing}")
                
            return data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
            
    def save_scaler(self, filepath: str) -> None:
        """Save fitted scaler to file"""
        if not hasattr(self.scaler, 'mean_'):
            raise ValueError("Scaler has not been fitted. Cannot save.")
            
        import joblib
        joblib.dump(self.scaler, filepath)
        
    @classmethod
    def load_scaler(cls, filepath: str) -> 'SupplyChainDataPreprocessor':
        """Load preprocessor with fitted scaler"""
        import joblib
        scaler = joblib.load(filepath)
        return cls(scaler=scaler) 