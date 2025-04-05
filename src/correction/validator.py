import hashlib
import os
import requests
from typing import Dict, List, Tuple

class PackageValidator:
    """Validates package integrity and dependencies"""
    
    def __init__(self, allowed_sources: List[str] = None):
        self.allowed_sources = allowed_sources or ["pypi.org", "github.com", "gitlab.com"]
        
    def validate_package(self, package_name: str, version: str) -> Tuple[bool, str]:
        """Validate a package's integrity and source"""
        try:
            # Check package source
            source_valid = self._validate_source(package_name)
            if not source_valid:
                return False, f"Package {package_name} is from an untrusted source"
            
            # Check package integrity
            integrity_valid = self._check_integrity(package_name, version)
            if not integrity_valid:
                return False, f"Package {package_name} version {version} failed integrity check"
            
            return True, "Package validation successful"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _validate_source(self, package_name: str) -> bool:
        """Check if package is from an allowed source"""
        try:
            # Query PyPI API for package info
            response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
            if response.status_code != 200:
                return False
                
            data = response.json()
            project_urls = data.get("info", {}).get("project_urls", {})
            
            # Check if any project URL is from allowed sources
            return any(
                source in str(url).lower() 
                for url in project_urls.values()
                for source in self.allowed_sources
            )
            
        except Exception:
            return False
    
    def _check_integrity(self, package_name: str, version: str) -> bool:
        """Verify package integrity using checksums"""
        try:
            # Get package info from PyPI
            response = requests.get(f"https://pypi.org/pypi/{package_name}/{version}/json")
            if response.status_code != 200:
                return False
                
            data = response.json()
            releases = data.get("releases", {}).get(version, [])
            
            if not releases:
                return False
                
            # Check if at least one release file has matching checksums
            for release in releases:
                if self._verify_checksums(release):
                    return True
                    
            return False
            
        except Exception:
            return False
    
    def _verify_checksums(self, release_info: Dict) -> bool:
        """Verify release file checksums"""
        try:
            md5_sum = release_info.get("md5_digest")
            sha256_sum = release_info.get("sha256_digest")
            
            # Download the file and verify checksums
            response = requests.get(release_info.get("url"), stream=True)
            if response.status_code != 200:
                return False
                
            # Calculate checksums
            md5_hash = hashlib.md5()
            sha256_hash = hashlib.sha256()
            
            for chunk in response.iter_content(chunk_size=8192):
                md5_hash.update(chunk)
                sha256_hash.update(chunk)
                
            return (
                md5_hash.hexdigest() == md5_sum and
                sha256_hash.hexdigest() == sha256_sum
            )
            
        except Exception:
            return False 