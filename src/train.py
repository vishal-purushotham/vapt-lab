import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import logging
import os
import pickle

from data_loader import load_and_prepare_data
from model import MTAD_GAT
from utils import normalize_data, SlidingWindowDataset, create_data_loaders
from trainer import Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration values (can be overridden by YAML and args)
DEFAULT_CONFIG = {
    'data': {
        'date': '2024-01-15', # Example date
        'start_time': '00:00:00',
        'end_time': '23:59:59',
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
        'window_size': 12, # e.g., 12 * 5min = 1 hour window
        'n_features': None, # Will be derived from data
        'hidden_size': 100,
        'fc_dropout': 0.3,
        'gru_dropout': 0.3,
        'gru_layers': 1,
        'use_gatv2': True,
        'use_bias': True,
        'target_dims': None # Optional: specify indices of features to target
    },
    'training': {
        'epochs': 50,
        'batch_size': 128,
        'init_lr': 0.001,
        'val_split': 0.1,
        'use_cuda': True,
        'output_dir': './output',
        'model_save_path': 'models',
        'log_dir': 'logs',
        'scaler_file': 'scaler.pkl'
    }
}

def main(args):
    # --- 1. Load Configuration ---
    config = DEFAULT_CONFIG.copy()
    try:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        # Deep merge YAML config into defaults
        # (A simple update might suffice depending on YAML structure)
        # This is a basic merge, consider a proper deep merge library for complex configs
        for key, value in yaml_config.items():
            if key in config and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value
        logger.info(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        logger.warning(f"Config file {args.config} not found. Using default/CLI arguments.")
    except Exception as e:
        logger.error(f"Error loading config file {args.config}: {e}. Using default/CLI args.")

    # Override with command-line arguments where provided
    config['data']['date'] = args.date if args.date else config['data']['date']
    config['data']['start_time'] = args.start_time if args.start_time else config['data']['start_time']
    config['data']['end_time'] = args.end_time if args.end_time else config['data']['end_time']
    config['training']['epochs'] = args.epochs if args.epochs else config['training']['epochs']
    config['training']['batch_size'] = args.batch_size if args.batch_size else config['training']['batch_size']
    config['training']['init_lr'] = args.lr if args.lr else config['training']['init_lr']
    config['training']['use_cuda'] = args.use_cuda if args.use_cuda is not None else config['training']['use_cuda']
    config['training']['output_dir'] = args.output_dir if args.output_dir else config['training']['output_dir']

    # Construct full paths for output files/dirs
    output_dir = config['training']['output_dir']
    model_save_dir = os.path.join(output_dir, config['training']['model_save_path'])
    log_dir = os.path.join(output_dir, config['training']['log_dir'])
    scaler_path = os.path.join(output_dir, config['training']['scaler_file'])
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger.info(f"Effective Configuration: {config}")

    # --- 2. Load and Prepare Data ---
    logger.info("Loading and preparing data...")
    wazuh_conf = config.get('wazuh', {})
    if not wazuh_conf.get('host') or not wazuh_conf.get('port') or not wazuh_conf.get('auth'):
         logger.error("Wazuh connection details missing in config ('wazuh' section with 'host', 'port', 'auth').")
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

    if data_df is None or data_df.empty:
        logger.error("No data loaded, exiting.")
        return

    logger.info(f"Data loaded successfully. Shape: {data_df.shape}")

    # Select only the numeric aggregated columns for normalization
    numeric_cols_tuples = data_df.select_dtypes(include=np.number).columns.tolist()
    if not isinstance(data_df.columns, pd.MultiIndex):
        logger.warning("Expected MultiIndex columns after aggregation, but found single-level index. Attempting to proceed.")
    numeric_cols_str = [f"{col[0]}_{col[1]}" for col in numeric_cols_tuples]
    logger.info(f"Numeric columns (tuples) to be normalized: {numeric_cols_tuples}")
    logger.info(f"Numeric columns (strings) to be saved: {numeric_cols_str}")

    if data_df.shape[0] <= config['model']['window_size']:
        logger.error("Not enough data for the specified window size. Exiting.")
        return

    # --- 3. Normalize Data ---
    logger.info("Normalizing data...")
    normalized_data, scaler = normalize_data(data_df[numeric_cols_tuples].values)

    # Save the scaler
    try:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved to {scaler_path}")
    except Exception as e:
        logger.error(f"Failed to save scaler: {e}")

    # --- 4. Split Data ---
    n_samples = normalized_data.shape[0]
    split_ratio = config['training']['val_split']
    split_idx = int(n_samples * (1 - split_ratio))

    # Ensure validation set is not empty if possible
    if split_idx == n_samples and n_samples > 0: # All data is training
        split_idx -= 1 # Assign at least one sample to validation if data exists

    train_data = normalized_data[:split_idx]
    val_data = normalized_data[split_idx:]

    if train_data.shape[0] == 0 or val_data.shape[0] == 0:
        logger.error(f"Validation split {split_ratio} results in an empty train or validation set.")
        logger.error(f"Total samples: {n_samples}, Train samples: {train_data.shape[0]}, Val samples: {val_data.shape[0]}")
        return

    logger.info(f"Data split into training ({train_data.shape[0]} samples) and validation ({val_data.shape[0]} samples)")

    # --- 5. Create Datasets and DataLoaders ---
    logger.info("Creating datasets and dataloaders...")
    window_size = config['model']['window_size']
    batch_size = config['training']['batch_size']

    # Check if enough data for at least one training sequence
    if train_data.shape[0] < window_size:
        logger.error(f"Training data length ({train_data.shape[0]}) is not sufficient for window size ({window_size}). Cannot create training sequences.")
        return

    train_dataset = SlidingWindowDataset(train_data, window_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    logger.info(f"Created training DataLoader with {len(train_dataset)} sequences.")

    # Check if enough data for validation sequences
    val_loader = None
    if val_data.shape[0] >= window_size:
        val_dataset = SlidingWindowDataset(val_data, window_size)
        # Ensure batch_size isn't larger than validation dataset size
        val_batch_size = min(batch_size, len(val_dataset))
        if val_batch_size > 0:
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
            logger.info(f"Created validation DataLoader with {len(val_dataset)} sequences.")
        else:
             # This case should technically not be hit if val_data.shape[0] >= window_size, but added for safety
            logger.warning(f"Validation data length ({val_data.shape[0]}) allows for sequences, but results in 0 samples for DataLoader? Skipping validation.")
    else:
        logger.warning(f"Validation data length ({val_data.shape[0]}) is less than window size ({window_size}). Skipping validation.")

    # --- 6. Initialize Model ---
    logger.info("Initializing model...")
    n_features = normalized_data.shape[1]
    config['model']['n_features'] = n_features # Update config with actual feature count

    model_params = config['model']
    model = MTAD_GAT(
        n_features=n_features,
        window_size=model_params['window_size'],
        out_dim=n_features, # Output dim matches input features for reconstruction
        kernel_size=model_params.get('kernel_size', 7), # Default from original model if not in config
        feat_gat_embed_dim=model_params.get('feat_gat_embed_dim'),
        time_gat_embed_dim=model_params.get('time_gat_embed_dim'),
        gru_n_layers=model_params.get('gru_layers', 1),
        gru_hid_dim=model_params.get('gru_hid_dim', 150), # Corrected lookup key
        forecast_n_layers=model_params.get('forecast_n_layers', 1),
        forecast_hid_dim=model_params.get('forecast_hid_dim', 150),
        recon_n_layers=model_params.get('recon_n_layers', 1),
        recon_hid_dim=model_params.get('recon_hid_dim', 150),
        dropout=model_params.get('dropout', 0.2),
        alpha=model_params.get('alpha', 0.2),
        use_gatv2=model_params.get('use_gatv2', True)
    )
    logger.info("Model initialized.")

    # --- 7. Initialize Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=config['training']['init_lr'])

    # --- 8. Initialize Trainer ---
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        window_size=model_params['window_size'],
        n_features=n_features,
        target_dims=model_params.get('target_dims'),
        n_epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        init_lr=config['training']['init_lr'],
        use_cuda=config['training']['use_cuda'],
        save_path=model_save_dir,
        log_dir=log_dir,
        args_summary=str(config) # Log the effective config
    )

    # --- 9. Start Training ---
    logger.info("Starting training process...")
    # Pass loaders to fit method (Trainer's fit method needs to handle val_loader being None)
    trainer.fit(train_loader, val_loader)

    logger.info("Training finished.")

    # --- 10. Save Final Configuration --- #
    config_save_path = os.path.join(output_dir, 'training_config.yaml')
    config['model']['n_features'] = n_features # Ensure n_features is in the saved config
    config['data']['numeric_columns'] = numeric_cols_str # Store the string representation of numeric columns
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    logger.info(f"Final training configuration saved to {config_save_path}")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MTAD-GAT model on Wazuh data.')

    # Configuration file
    parser.add_argument('--config', type=str, default='config/settings.yaml', # Corrected default path relative to project root
                        help='Path to the configuration YAML file.')

    # Data selection overrides
    parser.add_argument('--date', type=str, help='Date for training data (YYYY-MM-DD). Overrides config.')
    parser.add_argument('--start-time', type=str, help='Start time (HH:MM:SS). Overrides config.')
    parser.add_argument('--end-time', type=str, help='End time (HH:MM:SS). Overrides config.')

    # Training parameter overrides
    parser.add_argument('--epochs', type=int, help='Number of training epochs. Overrides config.')
    parser.add_argument('--batch-size', type=int, help='Batch size. Overrides config.')
    parser.add_argument('--lr', type=float, help='Initial learning rate. Overrides config.')
    parser.add_argument('--output-dir', type=str, help='Directory for output files (model, logs, scaler). Overrides config.')
    parser.add_argument('--use-cuda', action=argparse.BooleanOptionalAction, help='Enable/disable CUDA usage. Overrides config.')

    args = parser.parse_args()
    main(args)
