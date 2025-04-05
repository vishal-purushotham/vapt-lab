import os
import json
import shutil
import subprocess
from datetime import datetime
from typing import Dict, List, Optional

class PackageRollback:
    """Manages package version rollback and backup"""
    
    def __init__(self, backup_dir: str = "backups/", max_history: int = 5):
        self.backup_dir = backup_dir
        self.max_history = max_history
        self._ensure_backup_dir()
        
    def _ensure_backup_dir(self):
        """Create backup directory if it doesn't exist"""
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
            
    def backup_package(self, package_name: str, version: str) -> bool:
        """Create a backup of the current package state"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_info = {
                "package": package_name,
                "version": version,
                "timestamp": timestamp
            }
            
            # Save backup info
            backup_file = os.path.join(
                self.backup_dir, 
                f"{package_name}_{timestamp}.json"
            )
            with open(backup_file, "w") as f:
                json.dump(backup_info, f, indent=2)
                
            # Clean old backups
            self._cleanup_old_backups(package_name)
            
            return True
            
        except Exception as e:
            print(f"Backup failed: {str(e)}")
            return False
            
    def rollback_package(self, package_name: str, target_version: Optional[str] = None) -> bool:
        """Rollback package to a specific version or last known good version"""
        try:
            if target_version:
                # Rollback to specific version
                return self._install_version(package_name, target_version)
            else:
                # Find and rollback to last known good version
                backup = self._get_last_backup(package_name)
                if backup:
                    return self._install_version(
                        package_name, 
                        backup.get("version")
                    )
            return False
            
        except Exception as e:
            print(f"Rollback failed: {str(e)}")
            return False
            
    def _install_version(self, package_name: str, version: str) -> bool:
        """Install a specific version of a package"""
        try:
            result = subprocess.run(
                ["pip", "install", f"{package_name}=={version}"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
            
        except Exception:
            return False
            
    def _get_last_backup(self, package_name: str) -> Optional[Dict]:
        """Get the most recent backup for a package"""
        try:
            backups = []
            for filename in os.listdir(self.backup_dir):
                if filename.startswith(package_name) and filename.endswith(".json"):
                    with open(os.path.join(self.backup_dir, filename)) as f:
                        backup = json.load(f)
                        backups.append(backup)
                        
            if backups:
                # Sort by timestamp and return most recent
                return sorted(
                    backups,
                    key=lambda x: x["timestamp"],
                    reverse=True
                )[0]
            return None
            
        except Exception:
            return None
            
    def _cleanup_old_backups(self, package_name: str):
        """Remove old backups exceeding max_history"""
        try:
            backups = []
            for filename in os.listdir(self.backup_dir):
                if filename.startswith(package_name) and filename.endswith(".json"):
                    filepath = os.path.join(self.backup_dir, filename)
                    backups.append((
                        filename,
                        os.path.getmtime(filepath)
                    ))
                    
            # Sort by modification time (newest first)
            backups.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old backups
            for filename, _ in backups[self.max_history:]:
                os.remove(os.path.join(self.backup_dir, filename))
                
        except Exception as e:
            print(f"Cleanup failed: {str(e)}")