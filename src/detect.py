import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import logging
import os
import pickle

from data_loader import load_and_prepare_data
from model import MTAD_GAT
from utils import normalize_data, SlidingWindowDataset # Re-use normalize_data (with loaded scaler), SlidingWindowDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default config (can be partially overridden by YAML and args)
# Should align with the structure used in train.py
DEFAULT_CONFIG = {
    'data': {
        'columns_config': {
            'timestamp': 'datetime',
            'rule.level': 'int',
            'agent.ip': 'string',
            'data.command': 'string',
            'data.srcip': 'string',
            'data.dstip': 'string',
            'data.protocol': 'string',
            'data.srcport': 'int',
            'data.dstport': 'int',
        },
        'aggregation_config': {
            'rule.level': ['mean', 'max', 'count'],
            'data.srcport': 'nunique',
            'data.dstport': 'nunique',
        },
        'aggregation_window': '5min',
        'time_column': 'timestamp'
    },
    'model': {
        'window_size': 12,
        'target_dims': None,
        # Parameters needed to reconstruct the model architecture
        'n_features': None, # Will be loaded from saved info or inferred
        'hidden_size': 150,
        'gru_layers': 1,
        'use_gatv2': True,
        'use_bias': True,
        'dropout': 0.3,
        'kernel_size': 7,
        'feat_gat_embed_dim': None,
        'time_gat_embed_dim': None,
        'fc_n_layers': 3,
        'fc_hid_dim': 150,
        'recon_n_layers': 1,
        'recon_hid_dim': 150,
        'alpha': 0.2
    },
    'detection': {
        'gamma': 1.0, # Weight for reconstruction error in anomaly score
        'use_cuda': True,
        'batch_size': 128, # Batch size for prediction
        'model_path': './output/models/best_model.pt', # Default path relative to script location
        'scaler_path': './output/scaler.pkl', # Default path relative to script location
        'config_save_path': './output/training_config.yaml' # Path where training config was saved
    }
}

def detect_anomalies(args):
    # --- 1. Load Configuration ---
    config = DEFAULT_CONFIG.copy()
    try:
        # First, try loading the *training* config to get model params
        train_config_path = args.config_save_path or config['detection']['config_save_path']
        if os.path.exists(train_config_path):
            with open(train_config_path, 'r') as f:
                 train_config = yaml.safe_load(f)
            # Merge training config (especially model params) into defaults
            # This ensures model architecture matches the saved weights
            for key, value in train_config.items():
                if key in config and isinstance(config[key], dict):
                    config[key].update(value)
                else:
                    config[key] = value
            logger.info(f"Loaded training configuration from {train_config_path} for model parameters.")
        else:
            logger.warning(f"Training config {train_config_path} not found. Relying on defaults/CLI for model params.")

        # Optionally load a separate detection config file (if needed for overrides)
        if args.config and os.path.exists(args.config):
             with open(args.config, 'r') as f:
                 detect_config = yaml.safe_load(f)
             # Deep merge detection config
             for key, value in detect_config.items():
                if key in config and isinstance(config[key], dict):
                    config[key].update(value)
                else:
                    config[key] = value
             logger.info(f"Loaded detection-specific config from {args.config}")

    except Exception as e:
        logger.error(f"Error loading configuration: {e}. Using defaults/CLI args.")

    # Override with command-line arguments
    config['data']['date'] = args.date if args.date else config['data'].get('date', None) # Date is required for detection
    config['data']['start_time'] = args.start_time if args.start_time else config['data'].get('start_time', '00:00:00')
    config['data']['end_time'] = args.end_time if args.end_time else config['data'].get('end_time', '23:59:59')
    config['detection']['model_path'] = args.model or config['detection']['model_path']
    config['detection']['scaler_path'] = args.scaler or config['detection']['scaler_path']
    config['detection']['threshold'] = args.threshold # Threshold is mandatory from CLI
    config['detection']['gamma'] = args.gamma if args.gamma is not None else config['detection']['gamma']
    config['detection']['use_cuda'] = args.use_cuda if args.use_cuda is not None else config['detection']['use_cuda']
    config['model']['window_size'] = args.window_size if args.window_size else config['model']['window_size']

    if config['data']['date'] is None:
        logger.error("Date for detection ('--date') must be provided.")
        return
    if config['detection']['threshold'] is None:
        logger.error("Anomaly threshold ('--threshold') must be provided.")
        return

    logger.info(f"Effective Detection Configuration: {config}")

    # --- 2. Load Scaler and Model ---
    logger.info("Loading scaler and model...")
    scaler_path = config['detection']['scaler_path']
    model_path = config['detection']['model_path']

    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f"Scaler loaded from {scaler_path}")
    except FileNotFoundError:
        logger.error(f"Scaler file not found at {scaler_path}. Cannot proceed.")
        return
    except Exception as e:
        logger.error(f"Error loading scaler: {e}")
        return

    # Determine device
    use_cuda = config['detection']['use_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using device: {device}")

    # Infer n_features from scaler if not in config
    if config['model'].get('n_features') is None:
        try:
             config['model']['n_features'] = scaler.n_features_in_
             logger.info(f"Inferred n_features={config['model']['n_features']} from scaler.")
        except AttributeError:
             logger.error("Could not infer n_features from scaler and not found in config. Please train first or provide in config.")
             return

    # Initialize model architecture based on config
    model_params = config['model']
    try:
        model = MTAD_GAT(
            n_features=model_params['n_features'],
            window_size=model_params['window_size'],
            out_dim=model_params['n_features'], # Reconstruction matches features
            kernel_size=model_params.get('kernel_size', 7),
            feat_gat_embed_dim=model_params.get('feat_gat_embed_dim'),
            time_gat_embed_dim=model_params.get('time_gat_embed_dim'),
            gru_n_layers=model_params.get('gru_layers', 1),
            gru_hid_dim=model_params.get('hidden_size', 150),
            fc_n_layers=model_params.get('fc_layers', 3),
            fc_hid_dim=model_params.get('hidden_size', 150),
            recon_n_layers=model_params.get('recon_layers', 1),
            recon_hid_dim=model_params.get('recon_hid_dim', 150),
            alpha=model_params.get('alpha', 0.2),
            dropout=model_params.get('dropout', 0.3)
            # Ensure all necessary params used during training are here
        )
    except KeyError as e:
        logger.error(f"Missing model parameter in configuration: {e}. Ensure training config was loaded or defaults are sufficient.")
        return

    # Load model state
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # Set model to evaluation mode
        logger.info(f"Model loaded from {model_path}")
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}. Cannot proceed.")
        return
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # --- 3. Load and Prepare Data ---
    logger.info(f"Loading and preparing data for {config['data']['date']}...")
    wazuh_conf = config.get('wazuh', {})
    if not wazuh_conf.get('host') or not wazuh_conf.get('port') or not wazuh_conf.get('auth'):
         logger.error("Wazuh connection details missing in config ('wazuh' section). Ensure training config was loaded or provide in detection config.")
         return

    data_df = load_and_prepare_data(
        os_host=wazuh_conf['host'],
        os_port=wazuh_conf['port'],
        os_auth=(wazuh_conf['auth']['user'], wazuh_conf['auth']['password']),
        date=config['data']['date'],
        start_time_str=config['data']['start_time'],
        end_time_str=config['data']['end_time'],
        columns_config=config['data']['columns_config'],
        aggregation_config=config['data']['aggregation_config'],
        aggregation_window=config['data']['aggregation_window'],
        time_column=config['data']['time_column'],
        use_ssl=wazuh_conf.get('use_ssl', True),
        verify_certs=wazuh_conf.get('verify_certs', False),
        ssl_show_warn=wazuh_conf.get('ssl_show_warn', False)
    )

    if data_df.empty:
        logger.error("Failed to load data or no data found for the specified period. Exiting.")
        return

    logger.info(f"Data loaded successfully. Shape: {data_df.shape}")

    # --- 4. Preprocess Data ---
    logger.info("Preprocessing data...")
    time_col = config['data']['time_column']
    # Ensure columns match the scaler's expected features
    try:
        # Scaler fitted on data_df[numeric_cols] during training
        numeric_cols = scaler.feature_names_in_
        data_to_normalize = data_df[numeric_cols].values
    except (AttributeError, KeyError) as e:
        logger.error(f"Error identifying/extracting numeric columns based on scaler: {e}. Trying select_dtypes.")
        # Fallback if scaler info is missing
        numeric_cols = data_df.select_dtypes(include=np.number).columns.tolist()
        if time_col in numeric_cols:
            numeric_cols.remove(time_col)
        if len(numeric_cols) != model_params['n_features']:
            logger.error(f"Number of numeric columns ({len(numeric_cols)}) does not match model's expected features ({model_params['n_features']}).")
            return
        data_to_normalize = data_df[numeric_cols].values

    # Normalize using the loaded scaler
    try:
        normalized_data = scaler.transform(data_to_normalize)
    except Exception as e:
        logger.error(f"Error applying scaler transform: {e}")
        return

    # Create sliding windows
    window_size = config['model']['window_size']
    dataset = SlidingWindowDataset(normalized_data, window_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config['detection']['batch_size'], shuffle=False)

    # --- 5. Predict and Calculate Anomaly Scores ---
    logger.info("Predicting and calculating anomaly scores...")
    all_preds = []
    all_recons = []
    all_actuals = [] # Store actual values corresponding to predictions/reconstructions

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            y_hat, _ = model(x) # Forecast

            # Reconstruction of the last point in the window (y)
            recon_x = torch.cat((x[:, 1:, :], y), dim=1)
            _, window_recon = model(recon_x)
            last_recon = window_recon[:, -1, :] # Get reconstruction of y

            all_preds.append(y_hat.detach().cpu().numpy())
            all_recons.append(last_recon.detach().cpu().numpy())
            all_actuals.append(y.squeeze(1).detach().cpu().numpy()) # y is the actual value for the predicted step

    if not all_preds:
        logger.warning("No predictions generated (check data length vs window size).")
        return

    preds_np = np.concatenate(all_preds, axis=0)
    recons_np = np.concatenate(all_recons, axis=0)
    actual_np = np.concatenate(all_actuals, axis=0)

    # Calculate anomaly score
    gamma = config['detection']['gamma']
    # Ensure shapes match, sometimes dimensions might be squeezed
    if actual_np.ndim == 1 and preds_np.ndim == 2:
         actual_np = actual_np.reshape(-1, 1) # Reshape if needed
         if actual_np.shape[1] != preds_np.shape[1]:
              logger.error(f"Shape mismatch between actual ({actual_np.shape}) and preds ({preds_np.shape}) after reshape.")
              return
    elif actual_np.shape != preds_np.shape:
        logger.error(f"Shape mismatch between actual ({actual_np.shape}) and preds ({preds_np.shape})")
        return
    if actual_np.shape != recons_np.shape:
        logger.error(f"Shape mismatch between actual ({actual_np.shape}) and recons ({recons_np.shape})")
        return

    error_forecast = np.sqrt(np.sum((preds_np - actual_np)**2, axis=1))
    error_recon = np.sqrt(np.sum((recons_np - actual_np)**2, axis=1))
    anomaly_scores = error_forecast + gamma * error_recon

    # Align scores with original timestamps
    # Scores correspond to data points starting from window_size
    score_timestamps = data_df.index[window_size:]

    if len(anomaly_scores) != len(score_timestamps):
        logger.warning(f"Length mismatch: {len(anomaly_scores)} scores, {len(score_timestamps)} timestamps after windowing. Adjusting...")
        min_len = min(len(anomaly_scores), len(score_timestamps))
        anomaly_scores = anomaly_scores[:min_len]
        score_timestamps = score_timestamps[:min_len]

    scores_df = pd.DataFrame({
        'timestamp': score_timestamps,
        'score': anomaly_scores
    })
    scores_df.set_index('timestamp', inplace=True)

    # --- 6. Detect Anomalies based on Threshold ---
    threshold = config['detection']['threshold']
    logger.info(f"Applying threshold: {threshold}")
    anomalies = scores_df[scores_df['score'] > threshold]

    # --- 7. Report Anomalies ---
    if not anomalies.empty:
        logger.warning(f"Detected {len(anomalies)} potential anomalies:")
        # Print detected anomalies (timestamp and score)
        print("\n--- Potential Anomalies Detected ---")
        print(anomalies)
        print("------------------------------------\n")
        # Optionally save anomalies to a file
        # anomalies.to_csv('anomalies.csv')
    else:
        logger.info("No anomalies detected above the threshold.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect anomalies using a pre-trained MTAD-GAT model.')

    # Configuration file (optional, overrides defaults)
    parser.add_argument('--config', type=str, help='Path to a detection-specific YAML configuration file.')

    # Required arguments
    parser.add_argument('--date', type=str, required=True, help='Date for detection data (YYYY-MM-DD).')
    parser.add_argument('--threshold', type=float, required=True, help='Anomaly score threshold.')

    # Paths (overrides defaults or config)
    parser.add_argument('--model', type=str, help='Path to the trained model file (.pt).')
    parser.add_argument('--scaler', type=str, help='Path to the saved scaler file (.pkl).')
    parser.add_argument('--config-save-path', type=str, help='Path to the saved training config YAML (for model params).')

    # Data time range overrides (optional)
    parser.add_argument('--start-time', type=str, help='Start time (HH:MM:SS). Overrides config.')
    parser.add_argument('--end-time', type=str, help='End time (HH:MM:SS). Overrides config.')

    # Model/Detection parameter overrides (optional)
    parser.add_argument('--window-size', type=int, help='Sliding window size (must match training). Overrides config.')
    parser.add_argument('--gamma', type=float, help='Weight for reconstruction error in anomaly score. Overrides config.')
    parser.add_argument('--use-cuda', action=argparse.BooleanOptionalAction, help='Enable/disable CUDA usage. Overrides config.')

    args = parser.parse_args()
    detect_anomalies(args)
