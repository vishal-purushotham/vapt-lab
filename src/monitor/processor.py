import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class MetricsProcessor:
    """Processes raw monitoring data for the detector"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
    def process_metrics(self, metrics: List[Dict]) -> pd.DataFrame:
        """Convert raw metrics into a DataFrame"""
        processed_data = []
        
        for metric in metrics:
            record = {
                "timestamp": datetime.fromisoformat(metric["timestamp"]),
                "package_name": metric["package_name"]
            }
            
            # Flatten metrics
            for key, value in metric["metrics"].items():
                if key == "dependencies":
                    record["dependency_count"] = len(value)
                elif key == "size":
                    record["package_size"] = value
                else:
                    record[key] = value
                    
            processed_data.append(record)
            
        return pd.DataFrame(processed_data)
        
    def prepare_detector_features(
        self, 
        df: pd.DataFrame,
        required_columns: List[str]
    ) -> Tuple[np.ndarray, List[datetime]]:
        """Prepare features for the detector"""
        # Ensure all required columns exist
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0
                
        # Sort by timestamp
        df = df.sort_values("timestamp")
        timestamps = df["timestamp"].tolist()
        
        # Extract features
        features = df[required_columns].values
        
        # Create sequences
        sequences = []
        sequence_timestamps = []
        
        for i in range(len(features) - self.window_size + 1):
            seq = features[i:i + self.window_size]
            sequences.append(seq)
            sequence_timestamps.append(timestamps[i + self.window_size - 1])
            
        return np.array(sequences), sequence_timestamps
        
    def calculate_update_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate package update frequency"""
        df = df.copy()
        
        # Sort by timestamp
        df = df.sort_values("timestamp")
        
        # Calculate time difference between version changes
        df["version_changed"] = df["version"] != df["version"].shift(1)
        df["time_since_last_update"] = df["timestamp"].diff()
        
        # Calculate update frequency (updates per day)
        window = "1D"  # 1 day window
        df["update_frequency"] = (
            df["version_changed"]
            .rolling(window=window, on="timestamp")
            .sum()
            .fillna(0)
        )
        
        return df
        
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for better detection"""
        df = df.copy()
        
        # Calculate size changes
        df["size_change"] = df["package_size"].pct_change().fillna(0)
        
        # Calculate dependency volatility
        df["dependency_volatility"] = (
            df["dependency_count"]
            .rolling(window=5)
            .std()
            .fillna(0)
        )
        
        # Calculate resource usage intensity
        if "cpu_usage" in df.columns and "memory_usage" in df.columns:
            df["resource_intensity"] = (
                df["cpu_usage"] * 0.5 + 
                df["memory_usage"] * 0.5
            ).fillna(0)
            
        return df 