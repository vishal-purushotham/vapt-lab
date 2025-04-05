# Supply Chain Attack Detection System

A Python-based system for detecting and preventing supply chain attacks by monitoring package behavior. The system uses machine learning to detect anomalies and integrates with Wazuh for visualization and alerting.

## What Does This System Do?

This system helps you:
1. Monitor Python packages for suspicious behavior
2. Detect potential supply chain attacks
3. Automatically respond to threats
4. Visualize package behavior and alerts

## Complete Setup Guide

### Step 1: Prerequisites

1. Install Python 3.8 or higher
2. Install Wazuh (for visualization and alerts):
   ```bash
   # Download Wazuh installation script
   curl -sO https://packages.wazuh.com/4.7/wazuh-install.sh
   
   # Install Wazuh (includes dashboard)
   sudo bash wazuh-install.sh -a
   
   # Get your Wazuh credentials (save these)
   sudo cat /etc/wazuh-indexer/admin.pem
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Step 2: Configure Wazuh Integration

1. Create Wazuh credentials file:
   ```bash
   # Create config directory if it doesn't exist
   mkdir -p config
   
   # Create credentials file
   cat > config/wazuh_credentials.json << EOF
   {
       "host": "localhost",
       "port": 55000,
       "username": "wazuh",
       "password": "your-password-here"
   }
   EOF
   ```

2. Test Wazuh connection:
   ```bash
   python src/integration/test_connection.py
   ```

### Step 3: Set Up Package Monitoring

1. Create required directories:
   ```bash
   mkdir -p data/raw data/results
   ```

2. Start monitoring a package (e.g., numpy):
   ```bash
   python src/monitor/collector.py --package numpy --duration 3600
   ```

This will collect package metrics for 1 hour and save them to `data/raw/numpy_metrics.csv`

### Step 4: Run the Detection System

1. Configure detection settings:
   ```bash
   # Copy default config
   cp config/detector_config.yaml.example config/detector_config.yaml
   ```

2. Run detection on collected data:
   ```bash
   python src/detector/main.py --input data/raw/numpy_metrics.csv
   ```

### Step 5: Enable Protection

1. Start the correction handler (for package rollback):
   ```bash
   python src/correction/handler.py --watch
   ```

2. Start the mitigation handler (for threat response):
   ```bash
   python src/mitigation/handler.py --watch
   ```

### Step 6: View Results

1. Open Wazuh dashboard:
   ```
   https://localhost:443
   ```
   Login with your Wazuh credentials

2. Navigate to:
   - "Security Events" for alerts
   - "Package Monitoring" for metrics
   - "Anomaly Detection" for ML results

3. Or use Jupyter notebooks:
   ```bash
   jupyter notebook notebooks/analyze_results.ipynb
   ```

## Components

### 1. Detection (src/detector/)
- Identifies anomalous package behavior using ML models
- Processes package metrics and generates anomaly scores
- Example usage:
```bash
python src/detector/main.py --input data/raw/package_metrics.csv
```

### 2. Correction (src/correction/)
- Validates package integrity and dependencies
- Provides rollback capabilities for compromised packages
- Example usage:
```bash
python src/correction/handler.py --package numpy --validate
python src/correction/rollback.py --package numpy --version 1.21.0
```

### 3. Mitigation (src/mitigation/)
- Automatically responds to detected threats
- Blocks suspicious package updates
- Sends notifications for security events
- Example usage:
```bash
python src/mitigation/handler.py --config config/mitigation_config.yaml
```

### 4. Monitoring (src/monitor/)
- Collects real-time package metrics
- Tracks CPU, memory, and disk usage
- Monitors package update patterns
- Example usage:
```bash
python src/monitor/collector.py --package numpy --duration 3600
```

## Directory Structure

```
vaptlab/
├── src/                # Source code
│   ├── detector/      # Anomaly detection models
│   ├── correction/    # Package validation and rollback
│   ├── monitor/       # Metric collection and tracking
│   └── mitigation/    # Threat response and blocking
├── config/           # Component configurations
├── data/            # Data directory
│   ├── raw/        # Input datasets
│   └── results/    # Analysis results
├── notebooks/       # Analysis notebooks
└── tests/          # Unit tests
```

## Understanding the Output

### 1. Detection Results
File: `data/results/anomalies.csv`
```csv
timestamp,package,anomaly_score,is_anomaly
2024-01-01 00:10:00,numpy,0.85,True
```
- `anomaly_score`: 0-1 scale (higher = more suspicious)
- `is_anomaly`: True if score exceeds threshold

### 2. Monitoring Metrics
File: `data/results/metrics.csv`
```csv
timestamp,package,cpu_score,memory_score,overall_score
2024-01-01 00:10:00,numpy,0.75,0.95,0.85
```
- Tracks resource usage patterns
- Higher scores indicate unusual behavior

### 3. Wazuh Alerts
View in Wazuh dashboard:
- High Risk: Red alerts, immediate action needed
- Medium Risk: Yellow alerts, investigation needed
- Low Risk: Blue alerts, monitoring recommended

## Troubleshooting

Common issues and solutions:

1. Wazuh Connection Failed:
   ```bash
   # Check Wazuh status
   sudo systemctl status wazuh-manager
   # Check credentials in config/wazuh_credentials.json
   ```

2. Missing Data:
   - Ensure monitoring is running
   - Check data/raw directory for files
   - Verify file permissions

3. No Anomalies Detected:
   - Adjust threshold in config/detector_config.yaml
   - Increase monitoring duration
   - Check metric collection

4. System Requirements:
   - Minimum 4GB RAM
   - Python 3.8+
   - Wazuh 4.x+

## Getting Help

1. Check logs in `data/results/`
2. Review Wazuh dashboard for alerts
