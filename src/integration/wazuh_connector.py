import os
import json
import yaml
import requests
from typing import Dict, List, Optional
from datetime import datetime
from urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class WazuhConnector:
    """Connects the system with Wazuh for visualization and alerting"""
    
    def __init__(
        self,
        config_path: str = "config/wazuh_config.yaml",
        credentials_path: str = "config/wazuh_credentials.json"
    ):
        self.config = self._load_config(config_path)
        self.credentials = self._load_credentials(credentials_path)
        self.base_url = f"https://{self.credentials['host']}:{self.credentials['port']}"
        self.auth_header = self._get_auth_header()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load Wazuh integration configuration"""
        if not os.path.exists(config_path):
            return {
                "index_pattern": "vaptlab-*",
                "data_stream": {
                    "name": "vaptlab-detector",
                    "template": "vaptlab-detector-template"
                },
                "alert_levels": {
                    "high_risk": 12,
                    "medium_risk": 8,
                    "low_risk": 4
                }
            }
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _load_credentials(self, credentials_path: str) -> Dict:
        """Load Wazuh API credentials"""
        if not os.path.exists(credentials_path):
            return {
                "host": "localhost",
                "port": 55000,
                "username": "wazuh-wui",
                "password": "wazuh-wui"
            }
            
        with open(credentials_path, 'r') as f:
            return json.load(f)
            
    def _get_auth_header(self) -> Dict[str, str]:
        """Get authentication header for Wazuh API"""
        auth_endpoint = f"{self.base_url}/security/user/authenticate"
        
        try:
            response = requests.get(
                auth_endpoint,
                auth=(self.credentials['username'], self.credentials['password']),
                verify=False
            )
            response.raise_for_status()
            token = response.json()['data']['token']
            return {'Authorization': f'Bearer {token}'}
            
        except Exception as e:
            print(f"Failed to authenticate with Wazuh API: {str(e)}")
            return {}
            
    def create_index_pattern(self) -> bool:
        """Create index pattern for detector data"""
        endpoint = f"{self.base_url}/elastic/setup"
        
        try:
            data = {
                "pattern": self.config["index_pattern"],
                "time_field": "timestamp"
            }
            
            response = requests.post(
                endpoint,
                headers=self.auth_header,
                json=data,
                verify=False
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            print(f"Failed to create index pattern: {str(e)}")
            return False
            
    def create_data_stream(self) -> bool:
        """Create data stream for detector output"""
        endpoint = f"{self.base_url}/elastic/indices"
        
        try:
            data = {
                "name": self.config["data_stream"]["name"],
                "template": self.config["data_stream"]["template"],
                "mappings": {
                    "properties": {
                        "timestamp": {"type": "date"},
                        "package_name": {"type": "keyword"},
                        "anomaly_score": {"type": "float"},
                        "risk_level": {"type": "keyword"},
                        "actions_taken": {"type": "keyword"}
                    }
                }
            }
            
            response = requests.post(
                endpoint,
                headers=self.auth_header,
                json=data,
                verify=False
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            print(f"Failed to create data stream: {str(e)}")
            return False
            
    def send_alert(
        self,
        package_name: str,
        anomaly_score: float,
        risk_level: str,
        actions: List[str]
    ) -> bool:
        """Send alert to Wazuh"""
        endpoint = f"{self.base_url}/alerts"
        
        try:
            alert_data = {
                "timestamp": datetime.now().isoformat(),
                "rule": {
                    "level": self.config["alert_levels"][risk_level],
                    "description": f"Supply chain attack detected in {package_name}"
                },
                "data": {
                    "package_name": package_name,
                    "anomaly_score": anomaly_score,
                    "risk_level": risk_level,
                    "actions_taken": actions
                }
            }
            
            response = requests.post(
                endpoint,
                headers=self.auth_header,
                json=alert_data,
                verify=False
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            print(f"Failed to send alert: {str(e)}")
            return False
            
    def create_dashboard(self, name: str, description: str) -> bool:
        """Create a dashboard for visualizing detector data"""
        endpoint = f"{self.base_url}/elastic/dashboards"
        
        try:
            dashboard_data = {
                "name": name,
                "description": description,
                "panels": [
                    {
                        "type": "visualization",
                        "name": "Anomaly Scores Over Time",
                        "visualization": {
                            "type": "line",
                            "params": {
                                "field": "anomaly_score",
                                "time_field": "timestamp"
                            }
                        }
                    },
                    {
                        "type": "visualization",
                        "name": "Risk Level Distribution",
                        "visualization": {
                            "type": "pie",
                            "params": {
                                "field": "risk_level"
                            }
                        }
                    },
                    {
                        "type": "visualization",
                        "name": "Recent Alerts",
                        "visualization": {
                            "type": "table",
                            "params": {
                                "fields": [
                                    "timestamp",
                                    "package_name",
                                    "anomaly_score",
                                    "risk_level",
                                    "actions_taken"
                                ]
                            }
                        }
                    }
                ]
            }
            
            response = requests.post(
                endpoint,
                headers=self.auth_header,
                json=dashboard_data,
                verify=False
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            print(f"Failed to create dashboard: {str(e)}")
            return False 