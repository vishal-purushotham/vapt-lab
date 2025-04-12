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

    if data_df.empty:
        logger.error("Failed to load data or no data found for the specified period. Exiting.")
        return

    logger.info(f"Data loaded successfully. Shape: {data_df.shape}")

    # Identify numeric columns for scaling (excluding the time column)
    time_col = config['data']['time_column']
    numeric_cols = data_df.select_dtypes(include=np.number).columns.tolist()
    if time_col in numeric_cols:
        numeric_cols.remove(time_col)

    if not numeric_cols:
         logger.error("No numeric columns found in the aggregated data (excluding time column). Cannot proceed.")
         return

    logger.info(f"Numeric columns to be normalized: {numeric_cols}")
    data_to_normalize = data_df[numeric_cols].values

    # --- 3. Normalize Data ---
    logger.info("Normalizing data...")
    normalized_data, scaler = normalize_data(data_to_normalize)

    # Save the scaler
    try:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved to {scaler_path}")
    except Exception as e:
        logger.error(f"Error saving scaler: {e}")

    # --- 4. Create Datasets and DataLoaders ---
    logger.info("Creating datasets and dataloaders...")
    window_size = config['model']['window_size']
    batch_size = config['training']['batch_size']
    val_split = config['training']['val_split']

    full_dataset = SlidingWindowDataset(normalized_data, window_size)
    train_loader, val_loader = create_data_loaders(full_dataset, batch_size, val_split, shuffle=True)

    # --- 5. Initialize Model ---
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
        gru_hid_dim=model_params.get('hidden_size', 150),
        fc_n_layers=model_params.get('fc_layers', 3),
        fc_hid_dim=model_params.get('hidden_size', 150),
        recon_n_layers=model_params.get('recon_layers', 1),
        recon_hid_dim=model_params.get('recon_hid_dim', 150),
        alpha=model_params.get('alpha', 0.2),
        dropout=model_params.get('dropout', 0.3)
    )

    # --- 6. Initialize Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=config['training']['init_lr'])

    # --- 7. Initialize Trainer ---
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

    # --- 8. Start Training ---
    logger.info("Starting training process...")
    trainer.fit(train_loader, val_loader)

    logger.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MTAD-GAT model on Wazuh data.')

    # Configuration file
    parser.add_argument('--config', type=str, default='../config/settings.yaml',
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
