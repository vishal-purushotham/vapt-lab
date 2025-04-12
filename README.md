# AI-Powered Supply Chain Attack Detection using Wazuh Data

This project implements an AI-based anomaly detection system aimed at identifying potential **supply chain attacks** by analyzing patterns in Wazuh SIEM alert data. It leverages the MTAD-GAT (Multivariate Time-Series Anomaly Detection with Graph Attention Networks) model to learn baseline system behavior from alert statistics and flag significant deviations that might indicate compromise.

## Overview

The system works by:

1.  **Fetching Data:** Connecting to a Wazuh (OpenSearch) instance to retrieve security alerts relevant to system activity (e.g., file integrity changes, process execution, network connections) for a specified date.
2.  **Aggregation:** Aggregating these raw alerts into fixed time windows (e.g., 5 minutes) based on configured statistics (e.g., mean/max/count of `rule.level`). This transforms discrete events into a continuous time-series representation of system state.
3.  **Training (`train` mode):**
    *   Learning the "normal" patterns of aggregated alert activity using the MTAD-GAT model.
    *   Saving the trained model (`.pt`), data scaler (`.pkl`), and the exact configuration used (`training_config.yaml`). **This `training_config.yaml` is critical for ensuring consistency during detection.**
4.  **Detection (`detect` mode):**
    *   Loading a pre-trained model, scaler, and its `training_config.yaml`.
    *   Fetching and aggregating data for a new date.
    *   Calculating an anomaly score for each time window, measuring how much the current activity deviates from the learned norm.
    *   Flagging time windows where the score exceeds a defined threshold, indicating potentially anomalous behavior potentially related to a supply chain compromise.

## Key Concepts

*   **Supply Chain Context:** The underlying assumption is that certain types of supply chain attacks (e.g., compromised software updates, malicious dependencies) will manifest as detectable anomalies in system logs/alerts monitored by Wazuh.
*   **Anomaly Score:** Represents how unusual the aggregated alert patterns are in a given time window compared to the learned baseline. Higher scores indicate greater deviation.
*   **Aggregation Window:** The time interval (e.g., `5min`) over which raw alerts are summarized. An anomaly score applies to the entire window.
*   **`training_config.yaml`:** Saved during training, this file captures the model architecture, features used, and scaling parameters, ensuring the detection process correctly interprets the trained model.

## Project Structure

```
vaptproject/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config/
│   └── settings.yaml         # Main config (Wazuh, Data Aggregation, Model)
├── src/
│   ├── main.py               # CLI entry point (train/detect)
│   ├── data_loader.py        # Wazuh data handling
│   ├── model.py              # MTAD-GAT model definition
│   ├── trainer.py            # Training loop
│   ├── train.py              # Training orchestration
│   ├── detect.py             # Detection orchestration
│   ├── correction.py         # Placeholder for response actions
│   └── utils.py              # Utilities
└── output/                   # Default output directory
    ├── models/               # Saved models (e.g., model_final.pt)
    ├── logs/                 # TensorBoard logs
    ├── scaler.pkl            # Saved data scaler
    └── training_config.yaml  # CRUCIAL: Config from training run
```

## Setup

1.  **Clone Repository:** `git clone <your-repo-url>` and `cd vaptproject`
2.  **Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows: venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```
3.  **Install Dependencies:** `pip install -r requirements.txt` (Ensure compatible Python & PyTorch versions).

## Wazuh Environment Integration

This project requires access to a running Wazuh Indexer (OpenSearch) instance.

**(Refer to the original README sections or Wazuh documentation for detailed setup if needed. Key step: Ensure the Wazuh Indexer allows external connections by setting `network.host: 0.0.0.0` in its configuration and restarting the service.)**

## Configuration (`config/settings.yaml`)

Update this file before running:

*   **`wazuh` section:** Set `host`, `port`, `auth` (user/password) for your Wazuh Indexer.
*   **`data` section:** Configure `columns_config` (fields to extract), `aggregation_config` (stats like mean, max, count), `aggregation_window`.
*   **`model` & `training` sections:** Define model hyperparameters and training settings.

## Usage

### 1. Training the Model

**Goal:** Learn normal system behavior from historical Wazuh alerts.

```bash
python src/main.py train --date <YYYY-MM-DD> [OPTIONS]
```

*   **Required:** `--date <YYYY-MM-DD>` (ensure sufficient data exists).
*   Outputs are saved to `--output-dir` (default: `./output`), including the critical `training_config.yaml`.

### 2. Detecting Anomalies

**Goal:** Identify potential supply chain attack indicators in new data.

```bash
python src/main.py detect --date <YYYY-MM-DD> --threshold <T> [OPTIONS]
```

*   **Required:** `--date <YYYY-MM-DD>`, `--threshold <T>`.
*   **Crucial Option:** `--config-save-path <path>` (path to `training_config.yaml` from the relevant training run, default: `./output/training_config.yaml`). Must match the loaded `--model` and `--scaler`.

**Example (using defaults from `./output`):**
```bash
# Train on 2025-04-11
python src/main.py train --date 2025-04-11 --epochs 10

# Detect on 2025-04-12 with threshold 0.6
python src/main.py detect --date 2025-04-12 --threshold 0.6
```

## Correlating Findings with Wazuh Dashboard

When the script flags an anomalous time window (e.g., `2025-04-12 16:45:00`):

1.  Go to your Wazuh Dashboard -> Discover tab.
2.  Filter the time range to that specific window (e.g., `16:45:00` to `16:49:59` for a 5min window).
3.  Examine the raw alerts to understand *what specific events* (file changes, process starts, network traffic) contributed to the high anomaly score. This helps investigate if it relates to a potential supply chain vector.

## Troubleshooting Common Issues

*   **YAML Error (`...constructor for tag...python/tuple...`)**: Fixed by converting column names to strings before saving `training_config.yaml`. Retrain if using older code.
*   **Training Error (`Not enough data...`)**: Insufficient data for `--date`. Choose a different date or adjust `model.window_size`.
*   **Detection Error (`RuntimeError: Error(s) in loading state_dict...`)**: Mismatch between loaded model and `training_config.yaml`. Ensure `--config-save-path` points to the correct file for the loaded `--model`.
*   **Detection Error (`Wazuh connection details missing...`)**: Config issue. Check `training_config.yaml` loading or base `config/settings.yaml`.
*   **Data Errors (`KeyError`, `Missing columns...`)**: Columns defined in config don't match fetched Wazuh data. Verify fields.
*   **Scaler Errors (`ValueError: feature mismatch...`)**: Data features during detection don't match the loaded scaler. Ensure config consistency.

## Next Steps / Improvements

*   **Refine Features:** Select/engineer features more specific to known supply chain attack indicators.
*   **Anomaly Explanation:** Automatically retrieve associated raw alerts.
*   **Response Actions:** Implement `src/correction.py` (e.g., alert SOC, trigger active response).
*   **Advanced Thresholding:** Use adaptive methods (e.g., POT).
*   **Evaluation:** Add metrics if labeled attack data is available.
*   **Output Formatting:** Save results to structured files (CSV/JSON).
