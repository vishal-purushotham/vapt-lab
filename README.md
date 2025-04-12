# AI-Powered Supply Chain Attack Detection

This project implements an AI-based anomaly detection system 
for identifying potential supply chain attacks using Wazuh SIEM alert data. It adapts the MTAD-GAT model.

## Features

*   Fetches and preprocesses Wazuh alert data from OpenSearch.
*   Aggregates alert data into time-series features.
*   Trains an MTAD-GAT (Multivariate Time-Series Anomaly Detection with Graph Attention Networks) model.
*   Detects anomalies in new Wazuh data using the trained model.
*   Provides a configurable framework via YAML settings.

## Project Structure

```
vaptproject/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config/                   # Configuration files
│   └── settings.yaml         # Main configuration (Wazuh, Data, Model, Training)
├── src/                      # Source code
│   ├── main.py               # Main entry point (train/detect modes)
│   ├── data_loader.py        # Wazuh data fetching and preparation
│   ├── model.py              # MTAD-GAT model definition
│   ├── trainer.py            # Training loop implementation
│   ├── train.py              # Orchestrates the training process
│   ├── detect.py             # Orchestrates the anomaly detection process
│   ├── correction.py         # Placeholder for anomaly correction/response
│   └── utils.py              # Utility functions (normalization, datasets)
├── data/                     # Sample data (if any)
│   └── sample_wazuh_alerts.json # Example Wazuh alert format
└── output/                   # Default directory for outputs (created automatically)
    ├── models/               # Saved trained models
    ├── logs/                 # TensorBoard logs
    └── scaler.pkl            # Saved data scaler
    └── training_config.yaml  # Saved config used for training
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd vaptproject
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you have a compatible PyTorch version installed (CPU or GPU depending on your hardware and `use_cuda` setting).* 
    *See [PyTorch installation instructions](https://pytorch.org/get-started/locally/).*

## Wazuh Installation and Configuration (WSL Example)

This project requires a running Wazuh installation (Indexer, Manager, Dashboard) to source alert data. These instructions assume Wazuh is installed within a WSL (Windows Subsystem for Linux) distribution.

1.  **Install Wazuh:** Follow the official Wazuh documentation for installation. A helper script `wazuh-install.sh` might be included in this repository for convenience (review and adapt as needed).
2.  **Find WSL IP Address:** After installation, find the IP address assigned to your WSL instance. Run this command within WSL:
    ```bash
    ip addr | grep 'eth0' | grep 'inet '
    ```
    Note the IP address (e.g., `172.22.252.217`). You will need this for the Dashboard, Agents, and the Python script configuration.
3.  **Configure Wazuh Indexer for External Access:** By default, the Wazuh Indexer might only listen on `127.0.0.1`. To allow connections from your Windows host (where the Python script runs), you need to change this:
    *   Edit the Indexer configuration file in WSL:
        ```bash
        sudo nano /etc/wazuh-indexer/opensearch.yml
        ```
    *   Find the line `#network.host:` or `network.host: 127.0.0.1`.
    *   Change or uncomment it to:
        ```yaml
        network.host: 0.0.0.0
        ```
    *   Save the file (Ctrl+X, Y, Enter in nano) and restart the Indexer service:
        ```bash
        sudo systemctl restart wazuh-indexer
        ```
    *   Verify the change (optional): `sudo ss -tulnp | grep ':9200'` should now show `0.0.0.0:9200` or `:::9200`.
4.  **Access Wazuh Dashboard:** Open `https://<YOUR_WSL_IP>` in your browser. Log in using the `admin` credentials obtained during Wazuh installation.
5.  **(Optional) Check Manager Listening Ports:** Ensure the manager is listening for agent connections (UDP 1514, TCP 1515) externally:
    ```bash
    sudo ss -tulnp | grep -E ':1514|:1515'
    ```
    If only `127.0.0.1` is shown, you may need to adjust the manager's `ossec.conf` for remote connections (refer to Wazuh documentation).

## Wazuh Agent Installation (Windows Host Example)

To feed data into Wazuh, install agents on the machines you want to monitor.

1.  **Open PowerShell as Administrator** on the Windows machine.
2.  **Run the installation command**, replacing `<YOUR_WSL_IP>` with the actual IP found earlier:
    ```powershell
    Invoke-WebRequest -Uri https://packages.wazuh.com/4.x/windows/wazuh-agent-latest.msi -OutFile $env:TEMP\wazuh-agent.msi; msiexec.exe /i $env:TEMP\wazuh-agent.msi /q WAZUH_MANAGER='<YOUR_WSL_IP>' WAZUH_REGISTRATION_SERVER='<YOUR_WSL_IP>'
    ```
    *(Alternatively, follow the "Deploy New Agent" steps within the Wazuh Dashboard for a potentially customized command.)*
3.  **Start the Wazuh Agent Service**:
    ```powershell
    NET START WazuhSvc
    ```
4.  **Verify Connection:** Check the "Agents" section in the Wazuh Dashboard. The new agent should appear as "Active" after a minute or two.
    *   **Troubleshooting:** If the agent shows as "Disconnected", check the agent log file `C:\Program Files (x86)\ossec-agent\ossec.log` for error messages. Common issues include incorrect manager IP, firewall blocking ports (1514/1515), or the manager service not running.

## Configuration (`config/settings.yaml`)

1.  **`wazuh` section:**
    *   Set `host` to your Wazuh Indexer's IP address (e.g., the WSL IP `172.22.252.217`).
    *   Update `port` (usually 9200), `auth` (`user`, `password`) with your Wazuh Indexer/OpenSearch credentials.
    *   Adjust `use_ssl` and `verify_certs` as needed for your OpenSearch setup (defaults usually work for standard Wazuh installs).
2.  **`data` section:**
    *   Configure `columns_config` with fields you want to extract from Wazuh alerts and their types. **Important:** Only include fields that *actually exist* in your alert data. Check sample alerts in the Wazuh Dashboard (Discover tab) if unsure.
    *   Define `aggregation_config` for how numeric features should be aggregated within the `aggregation_window`. Ensure columns listed here exist in `columns_config` and the data.
3.  **`model` section:**
    *   Adjust parameters like `window_size`, `hidden_size`, `dropout` if needed. `n_features` is determined automatically during training.
4.  **`training` section:**
    *   Set `epochs`, `batch_size`, `init_lr`, `val_split`, and `output_dir`.

## Usage

The main entry point is `src/main.py`, which supports two modes: `train` and `detect`.

### 1. Training the Model

**Prerequisite:** Ensure you have sufficient Wazuh alert data for the target training date, spanning a duration longer than `model.window_size * data.aggregation_window`.

Run the following command from the `vaptproject` directory:

```bash
python src/main.py train [OPTIONS]
```

**Key Options:**

*   `--config <path>`: Path to the YAML configuration file (default: `config/settings.yaml`).
*   `--date <YYYY-MM-DD>`: Override the date specified in the config for fetching training data. **Crucial: Choose a date with enough existing alert data.**
*   `--epochs <N>`: Override the number of training epochs.
*   `--output-dir <path>`: Override the output directory for models, logs, and scaler.
*   `--use-cuda / --no-use-cuda`: Explicitly enable/disable GPU usage.
*   `--help`: Show all available options.

**Example:**

```bash
# Train using settings from config/settings.yaml for date 2024-07-10
python src/main.py train --date 2024-07-10

# Train for 100 epochs, saving outputs to ./my_training_run
python src/main.py train --epochs 100 --output-dir ./my_training_run
```

Training outputs (model, scaler, logs, training config copy) will be saved in the specified `output_dir` (default: `./output`).

### 2. Detecting Anomalies

After training, use the `detect` mode to find anomalies in new data.

```bash
python src/main.py detect --date <YYYY-MM-DD> --threshold <T> [OPTIONS]
```

**Required Arguments:**

*   `--date <YYYY-MM-DD>`: The date for which to fetch data and detect anomalies.
*   `--threshold <T>`: The anomaly score threshold. Scores above this value will be flagged as anomalies.

**Key Options:**

*   `--model <path>`: Path to the trained model file (`.pt`). Defaults to the model in the `./output` directory structure.
*   `--scaler <path>`: Path to the saved scaler file (`.pkl`). Defaults to the scaler in the `./output` directory.
*   `--config-save-path <path>`: Path to the `training_config.yaml` saved during training (needed to reconstruct model architecture). Defaults to the one in `./output`.
*   `--config <path>`: Path to a *detection-specific* YAML config (can override Wazuh connection details, etc., if different from training).
*   `--gamma <G>`: Override the weight for the reconstruction error component of the anomaly score.
*   `--help`: Show all available options.

**Example:**

```bash
# Detect anomalies for 2024-07-11 using default model/scaler paths and threshold 5.0
python src/main.py detect --date 2024-07-11 --threshold 5.0

# Detect using a specific model and scaler with threshold 7.5
python src/main.py detect --date 2024-07-12 --threshold 7.5 --model ./my_training_run/models/best_model.pt --scaler ./my_training_run/scaler.pkl --config-save-path ./my_training_run/training_config.yaml
```

Detected anomalies (timestamps and scores) will be printed to the console.

## Next Steps / Improvements

*   **Correction Logic:** Implement actual correction/response actions in `src/correction.py` (e.g., Wazuh active response integration).
*   **Thresholding:** Implement more sophisticated thresholding methods (e.g., POT, epsilon from `idps-escape`) instead of a fixed threshold.
*   **Evaluation:** Add proper evaluation metrics (Precision, Recall, F1-Score) if labeled anomaly data is available.
*   **Data Handling:** Improve robustness of data loading and preprocessing.
*   **Feature Engineering:** Experiment with different features and aggregation methods.
*   **Hyperparameter Tuning:** Optimize model and training parameters.
*   **Error Handling:** Enhance error handling and logging.
