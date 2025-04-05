import os
import yaml
import json
import logging
import subprocess
from typing import Dict, List, Optional
from datetime import datetime

from ..correction.handler import CorrectionHandler
from ..detector.model import SupplyChainDetector

class MitigationHandler:
    """Handles automated response actions when threats are detected"""
    
    def __init__(
        self,
        config_path: str = "config/mitigation_config.yaml",
        correction_handler: Optional[CorrectionHandler] = None
    ):
        self.config = self._load_config(config_path)
        self.correction_handler = correction_handler or CorrectionHandler()
        self.logger = self._setup_logger()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load mitigation configuration"""
        if not os.path.exists(config_path):
            return {
                "response_actions": {
                    "high_risk": ["rollback", "block_updates", "notify"],
                    "medium_risk": ["validate", "notify"],
                    "low_risk": ["notify"]
                },
                "notification": {
                    "email": {
                        "enabled": False,
                        "recipients": []
                    },
                    "logging": {
                        "enabled": True,
                        "path": "logs/mitigation.log"
                    }
                },
                "thresholds": {
                    "high_risk": 0.8,
                    "medium_risk": 0.6,
                    "low_risk": 0.3
                }
            }
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for mitigation actions"""
        logger = logging.getLogger("mitigation")
        logger.setLevel(logging.INFO)
        
        if self.config["notification"]["logging"]["enabled"]:
            log_path = self.config["notification"]["logging"]["path"]
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            
            handler = logging.FileHandler(log_path)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def handle_threat(
        self,
        package_name: str,
        anomaly_score: float,
        detection_time: datetime
    ) -> List[str]:
        """Handle detected threats based on anomaly score"""
        # Determine risk level
        risk_level = self._determine_risk_level(anomaly_score)
        
        # Get response actions for risk level
        actions = self.config["response_actions"][risk_level]
        
        # Execute response actions
        executed_actions = []
        for action in actions:
            success = self._execute_action(action, package_name, anomaly_score)
            if success:
                executed_actions.append(action)
                
        # Log response
        self._log_response(
            package_name,
            risk_level,
            anomaly_score,
            executed_actions,
            detection_time
        )
        
        return executed_actions
        
    def _determine_risk_level(self, anomaly_score: float) -> str:
        """Determine risk level based on anomaly score"""
        thresholds = self.config["thresholds"]
        
        if anomaly_score >= thresholds["high_risk"]:
            return "high_risk"
        elif anomaly_score >= thresholds["medium_risk"]:
            return "medium_risk"
        elif anomaly_score >= thresholds["low_risk"]:
            return "low_risk"
        else:
            return "low_risk"
            
    def _execute_action(
        self,
        action: str,
        package_name: str,
        anomaly_score: float
    ) -> bool:
        """Execute a specific response action"""
        try:
            if action == "rollback":
                return self.correction_handler.force_rollback(package_name)
                
            elif action == "validate":
                validation_result = self.correction_handler.validate_package(
                    package_name
                )
                return validation_result[0]  # Returns success boolean
                
            elif action == "block_updates":
                return self._block_package_updates(package_name)
                
            elif action == "notify":
                return self._send_notification(
                    package_name,
                    anomaly_score
                )
                
            else:
                self.logger.warning(f"Unknown action: {action}")
                return False
                
        except Exception as e:
            self.logger.error(
                f"Error executing action {action} for {package_name}: {str(e)}"
            )
            return False
            
    def _block_package_updates(self, package_name: str) -> bool:
        """Block future updates for a package"""
        try:
            # Create or update apt preferences file
            prefs_dir = "/etc/apt/preferences.d"
            os.makedirs(prefs_dir, exist_ok=True)
            
            prefs_file = os.path.join(prefs_dir, f"{package_name}-block")
            with open(prefs_file, 'w') as f:
                f.write(f"""Package: {package_name}
Pin: version *
Pin-Priority: -1
""")
            
            self.logger.info(f"Blocked updates for package: {package_name}")
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to block updates for {package_name}: {str(e)}"
            )
            return False
            
    def _send_notification(
        self,
        package_name: str,
        anomaly_score: float
    ) -> bool:
        """Send notifications about detected threats"""
        success = True
        
        # Log notification
        if self.config["notification"]["logging"]["enabled"]:
            self.logger.warning(
                f"Threat detected in package {package_name} "
                f"with anomaly score {anomaly_score:.3f}"
            )
            
        # Email notification
        if self.config["notification"]["email"]["enabled"]:
            try:
                recipients = self.config["notification"]["email"]["recipients"]
                subject = f"Security Alert: Threat Detected in {package_name}"
                body = (
                    f"A potential security threat was detected:\n\n"
                    f"Package: {package_name}\n"
                    f"Anomaly Score: {anomaly_score:.3f}\n"
                    f"Detection Time: {datetime.now().isoformat()}\n"
                )
                
                # Use mail command for simple email sending
                for recipient in recipients:
                    cmd = f'echo "{body}" | mail -s "{subject}" {recipient}'
                    subprocess.run(
                        cmd,
                        shell=True,
                        check=True,
                        capture_output=True
                    )
                    
            except Exception as e:
                self.logger.error(f"Failed to send email notification: {str(e)}")
                success = False
                
        return success
        
    def _log_response(
        self,
        package_name: str,
        risk_level: str,
        anomaly_score: float,
        executed_actions: List[str],
        detection_time: datetime
    ):
        """Log the response actions taken"""
        log_entry = {
            "timestamp": detection_time.isoformat(),
            "package": package_name,
            "risk_level": risk_level,
            "anomaly_score": anomaly_score,
            "actions_taken": executed_actions
        }
        
        self.logger.info(json.dumps(log_entry)) 