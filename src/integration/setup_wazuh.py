#!/usr/bin/env python3

import os
import sys
import argparse
from wazuh_connector import WazuhConnector

def setup_wazuh_integration(
    config_path: str = "config/wazuh_config.yaml",
    credentials_path: str = "config/wazuh_credentials.json"
) -> bool:
    """Set up Wazuh integration for supply chain attack detection"""
    
    print("Setting up Wazuh integration...")
    
    # Initialize connector
    try:
        connector = WazuhConnector(config_path, credentials_path)
    except Exception as e:
        print(f"Failed to initialize Wazuh connector: {str(e)}")
        return False
        
    # Create index pattern
    print("Creating index pattern...")
    if not connector.create_index_pattern():
        print("Failed to create index pattern")
        return False
        
    # Create data stream
    print("Creating data stream...")
    if not connector.create_data_stream():
        print("Failed to create data stream")
        return False
        
    # Create default dashboard
    print("Creating default dashboard...")
    if not connector.create_dashboard(
        "Supply Chain Attack Detection",
        "Dashboard for monitoring supply chain attacks"
    ):
        print("Failed to create dashboard")
        return False
        
    print("Wazuh integration setup completed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Set up Wazuh integration for supply chain attack detection"
    )
    
    parser.add_argument(
        "--config",
        default="config/wazuh_config.yaml",
        help="Path to Wazuh configuration file"
    )
    
    parser.add_argument(
        "--credentials",
        default="config/wazuh_credentials.json",
        help="Path to Wazuh credentials file"
    )
    
    args = parser.parse_args()
    
    # Check if config files exist
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        sys.exit(1)
        
    if not os.path.exists(args.credentials):
        print(f"Credentials file not found: {args.credentials}")
        sys.exit(1)
        
    # Run setup
    success = setup_wazuh_integration(args.config, args.credentials)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 