import os
import json
import time
import numpy as np
from datetime import datetime
from typing import List, Optional

from .collector import PackageMonitor
from .processor import MetricsProcessor

class MonitoringManager:
    """Manages package monitoring and data processing"""
    
    def __init__(
        self,
        output_dir: str = "monitoring_data",
        check_interval: int = 300,
        window_size: int = 100
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.monitor = PackageMonitor(check_interval=check_interval)
        self.processor = MetricsProcessor(window_size=window_size)
        
        # Required features for the detector
        self.required_features = [
            "package_size",
            "dependency_count",
            "update_frequency",
            "size_change",
            "dependency_volatility",
            "resource_intensity"
        ]
        
    def monitor_packages(
        self,
        packages: List[str],
        duration: int = 3600,
        save_raw: bool = True
    ):
        """Monitor specified packages and process the data"""
        for package in packages:
            print(f"Starting monitoring for package: {package}")
            
            # Monitor package
            metrics = self.monitor.monitor_package(
                package_name=package,
                duration=duration
            )
            
            # Save raw metrics if requested
            if save_raw:
                raw_file = os.path.join(
                    self.output_dir,
                    f"{package}_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                with open(raw_file, "w") as f:
                    json.dump(metrics, f, indent=2)
                    
            # Process metrics
            df = self.processor.process_metrics(metrics)
            
            # Add derived features
            df = self.processor.calculate_update_frequency(df)
            df = self.processor.add_derived_features(df)
            
            # Prepare sequences for detector
            sequences, timestamps = self.processor.prepare_detector_features(
                df, self.required_features
            )
            
            # Save processed data
            processed_file = os.path.join(
                self.output_dir,
                f"{package}_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
            )
            np.savez(
                processed_file,
                sequences=sequences,
                timestamps=timestamps,
                feature_names=self.required_features
            )
            
            print(f"Completed monitoring for package: {package}")
            print(f"Raw data saved to: {raw_file}")
            print(f"Processed data saved to: {processed_file}")
            
def main():
    """Main entry point for the monitoring component"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Package monitoring tool")
    parser.add_argument(
        "--packages",
        nargs="+",
        required=True,
        help="List of packages to monitor"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=3600,
        help="Monitoring duration in seconds (default: 3600)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval in seconds (default: 300)"
    )
    parser.add_argument(
        "--output",
        default="monitoring_data",
        help="Output directory (default: monitoring_data)"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Window size for sequence preparation (default: 100)"
    )
    
    args = parser.parse_args()
    
    manager = MonitoringManager(
        output_dir=args.output,
        check_interval=args.interval,
        window_size=args.window
    )
    
    try:
        manager.monitor_packages(
            packages=args.packages,
            duration=args.duration
        )
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
    except Exception as e:
        print(f"Error during monitoring: {str(e)}")
        
if __name__ == "__main__":
    main() 