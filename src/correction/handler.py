import os
import yaml
from typing import Dict, List, Optional, Tuple
from .validator import PackageValidator
from .rollback import PackageRollback

class CorrectionHandler:
    """Handles package validation and correction based on detector findings"""
    
    def __init__(self, config_path: str = "config/correction_config.yaml"):
        self.config = self._load_config(config_path)
        self.validator = PackageValidator(
            allowed_sources=self.config["validation"]["dependency_check"]["allowed_sources"]
        )
        self.rollback = PackageRollback(
            backup_dir=self.config["rollback"]["backup_dir"],
            max_history=self.config["rollback"]["max_history"]
        )
        
    def _load_config(self, config_path: str) -> Dict:
        """Load correction configuration"""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            return {
                "rollback": {"backup_dir": "backups/", "max_history": 5},
                "validation": {
                    "dependency_check": {
                        "allowed_sources": ["pypi.org", "github.com", "gitlab.com"]
                    }
                }
            }
            
    def handle_detection(self, package_name: str, version: str, anomaly_score: float) -> bool:
        """Handle detected anomalies in packages"""
        try:
            # If anomaly score is high enough, validate and potentially rollback
            if anomaly_score > self.config.get("detection", {}).get("threshold", 0.8):
                print(f"Suspicious package detected: {package_name} v{version}")
                
                # Validate package
                is_valid, message = self.validator.validate_package(package_name, version)
                if not is_valid:
                    print(f"Validation failed: {message}")
                    
                    # Attempt rollback
                    if self.rollback.rollback_package(package_name):
                        print(f"Successfully rolled back {package_name}")
                        return True
                    else:
                        print(f"Failed to rollback {package_name}")
                        return False
                        
                print(f"Package {package_name} passed validation")
                return True
                
            return True
            
        except Exception as e:
            print(f"Error handling detection: {str(e)}")
            return False
            
    def backup_current_state(self, package_name: str, version: str) -> bool:
        """Create backup of current package state"""
        return self.rollback.backup_package(package_name, version)
        
    def validate_package(self, package_name: str, version: str) -> Tuple[bool, str]:
        """Validate a package"""
        return self.validator.validate_package(package_name, version)
        
    def force_rollback(self, package_name: str, version: Optional[str] = None) -> bool:
        """Force package rollback"""
        return self.rollback.rollback_package(package_name, version) 