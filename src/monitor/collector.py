import os
import time
import psutil
import json
from datetime import datetime
from typing import Dict, List, Optional
import pkg_resources
import subprocess

class PackageMonitor:
    """Monitors package-related metrics and system state"""
    
    def __init__(self, check_interval: int = 300):
        self.check_interval = check_interval
        self.metrics_history = []
        
    def collect_package_metrics(self, package_name: str) -> Dict:
        """Collect metrics for a specific package"""
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "package_name": package_name,
                "metrics": {}
            }
            
            # Get package info
            pkg = pkg_resources.working_set.by_key.get(package_name)
            if pkg:
                metrics["metrics"].update({
                    "version": pkg.version,
                    "dependencies": [str(r) for r in pkg.requires()],
                    "location": pkg.location
                })
                
            # Get package size
            if pkg and os.path.exists(pkg.location):
                size = self._get_directory_size(pkg.location)
                metrics["metrics"]["size"] = size
                
            # Get process info if package is running
            proc_info = self._get_process_info(package_name)
            if proc_info:
                metrics["metrics"].update(proc_info)
                
            return metrics
            
        except Exception as e:
            print(f"Error collecting metrics: {str(e)}")
            return {}
            
    def _get_directory_size(self, path: str) -> int:
        """Calculate total size of a directory"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size
        
    def _get_process_info(self, package_name: str) -> Optional[Dict]:
        """Get process information for a package"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                if package_name.lower() in proc.info['name'].lower():
                    return {
                        "cpu_usage": proc.info['cpu_percent'],
                        "memory_usage": proc.info['memory_percent'],
                        "pid": proc.info['pid']
                    }
            return None
            
        except Exception:
            return None
            
    def monitor_package(self, package_name: str, duration: int = 3600) -> List[Dict]:
        """Monitor a package for a specified duration"""
        start_time = time.time()
        end_time = start_time + duration
        
        while time.time() < end_time:
            metrics = self.collect_package_metrics(package_name)
            if metrics:
                self.metrics_history.append(metrics)
                
            time.sleep(self.check_interval)
            
        return self.metrics_history
        
    def save_metrics(self, filepath: str):
        """Save collected metrics to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            return True
            
        except Exception as e:
            print(f"Error saving metrics: {str(e)}")
            return False
            
    def check_package_updates(self, package_name: str) -> Optional[Dict]:
        """Check for available package updates"""
        try:
            # Get current version
            pkg = pkg_resources.working_set.by_key.get(package_name)
            if not pkg:
                return None
                
            current_version = pkg.version
            
            # Check PyPI for latest version
            result = subprocess.run(
                ["pip", "index", "versions", package_name],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Parse output to get latest version
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Available versions:' in line:
                        versions = line.split(':')[1].strip().split(',')
                        if versions:
                            latest_version = versions[0].strip()
                            return {
                                "package": package_name,
                                "current_version": current_version,
                                "latest_version": latest_version,
                                "update_available": latest_version != current_version
                            }
                            
            return None
            
        except Exception as e:
            print(f"Error checking updates: {str(e)}")
            return None