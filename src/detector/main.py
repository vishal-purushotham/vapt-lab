import os
import yaml
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

from .model import SupplyChainDetector
from ..mitigation.handler import MitigationHandler
from ..integration.wazuh_connector import WazuhConnector

class DetectionManager:
    """Manages the detection process and integrates with mitigation and visualization"""
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "config/detector_config.yaml",
        mitigation_config: str = "config/mitigation_config.yaml",
        wazuh_config: str = "config/wazuh_config.yaml",
        wazuh_credentials: str = "config/wazuh_credentials.json"
    ):
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.detector = self._load_model(model_path)
        self.mitigation = MitigationHandler(mitigation_config)
        self.wazuh = WazuhConnector(wazuh_config, wazuh_credentials)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load detector configuration"""
        if not os.path.exists(config_path):
            return {
                "batch_size": 32,
                "sequence_length": 100,
                "feature_size": 10,
                "threshold": 0.5
            }
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _load_model(self, model_path: str) -> SupplyChainDetector:
        """Load the trained detector model"""
        model = SupplyChainDetector(
            input_size=self.config["feature_size"],
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        )
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
        
    def process_package(
        self,
        package_name: str,
        features: np.ndarray
    ) -> Dict:
        """Process a package and handle any detected threats"""
        # Prepare input
        features_tensor = torch.FloatTensor(features).to(self.device)
        features_tensor = features_tensor.unsqueeze(0)  # Add batch dimension
        
        # Get prediction
        with torch.no_grad():
            anomaly_score = self.detector(features_tensor)
            anomaly_score = anomaly_score.cpu().numpy()[0]
            
        # Handle threat if detected
        if anomaly_score > self.config["threshold"]:
            detection_time = datetime.now()
            
            # Execute mitigation actions
            actions = self.mitigation.handle_threat(
                package_name,
                anomaly_score,
                detection_time
            )
            
            # Send alert to Wazuh
            risk_level = self.mitigation._determine_risk_level(anomaly_score)
            self.wazuh.send_alert(
                package_name,
                anomaly_score,
                risk_level,
                actions
            )
            
            return {
                "timestamp": detection_time.isoformat(),
                "package_name": package_name,
                "anomaly_score": float(anomaly_score),
                "threshold": self.config["threshold"],
                "is_anomaly": True,
                "risk_level": risk_level,
                "actions_taken": actions
            }
            
        return {
            "timestamp": datetime.now().isoformat(),
            "package_name": package_name,
            "anomaly_score": float(anomaly_score),
            "threshold": self.config["threshold"],
            "is_anomaly": False,
            "risk_level": "none",
            "actions_taken": []
        }
        
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Supply chain attack detection system"
    )
    
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model weights"
    )
    
    parser.add_argument(
        "--config",
        default="config/detector_config.yaml",
        help="Path to detector configuration"
    )
    
    parser.add_argument(
        "--mitigation-config",
        default="config/mitigation_config.yaml",
        help="Path to mitigation configuration"
    )
    
    parser.add_argument(
        "--wazuh-config",
        default="config/wazuh_config.yaml",
        help="Path to Wazuh configuration"
    )
    
    parser.add_argument(
        "--wazuh-credentials",
        default="config/wazuh_credentials.json",
        help="Path to Wazuh credentials"
    )
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = DetectionManager(
        model_path=args.model,
        config_path=args.config,
        mitigation_config=args.mitigation_config,
        wazuh_config=args.wazuh_config,
        wazuh_credentials=args.wazuh_credentials
    )
    
    # Example usage
    print("Detection system initialized and ready!")
    print("Use the process_package() method to analyze packages.")
    
if __name__ == "__main__":
    main()