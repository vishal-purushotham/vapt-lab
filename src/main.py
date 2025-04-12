import argparse
import logging
import sys
import os

# Add src directory to Python path to allow sibling imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions from other modules AFTER adjusting path
# We wrap these imports in try-except blocks to allow running --help
# even if dependencies are not yet installed.
try:
    from train import main as run_training
    from detect import detect_anomalies
    # from correction import apply_correction_measures # Import if needed later
except ImportError as e:
    # Allow basic arg parsing even if imports fail (e.g., for --help)
    if '--help' not in sys.argv and '-h' not in sys.argv:
        print(f"Error importing modules: {e}", file=sys.stderr)
        print("Please ensure all dependencies are installed (pip install -r requirements.txt) and scripts are in the correct place.", file=sys.stderr)
        sys.exit(1)
    # Define placeholder functions if imports fail but help is requested
    def run_training(args): pass
    def detect_anomalies(args): pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='AI Supply Chain Attack Detection Tool')

    subparsers = parser.add_subparsers(dest='mode', required=True,
                                         help='Select mode: train or detect')

    # --- Train Mode Arguments ---
    parser_train = subparsers.add_parser('train', help='Train the anomaly detection model')
    # Configuration file
    parser_train.add_argument('--config', type=str, default='../config/settings.yaml',
                              help='Path to the configuration YAML file.')
    # Data selection overrides
    parser_train.add_argument('--date', type=str, help='Date for training data (YYYY-MM-DD). Overrides config.')
    parser_train.add_argument('--start-time', type=str, help='Start time (HH:MM:SS). Overrides config.')
    parser_train.add_argument('--end-time', type=str, help='End time (HH:MM:SS). Overrides config.')
    # Training parameter overrides
    parser_train.add_argument('--epochs', type=int, help='Number of training epochs. Overrides config.')
    parser_train.add_argument('--batch-size', type=int, help='Batch size. Overrides config.')
    parser_train.add_argument('--lr', type=float, help='Initial learning rate. Overrides config.')
    parser_train.add_argument('--output-dir', type=str, help='Directory for output files (model, logs, scaler). Overrides config.')
    parser_train.add_argument('--use-cuda', action=argparse.BooleanOptionalAction, help='Enable/disable CUDA usage. Overrides config.')
    parser_train.set_defaults(func=run_training)

    # --- Detect Mode Arguments ---
    parser_detect = subparsers.add_parser('detect', help='Detect anomalies using a trained model')
    # Configuration file (optional, detection specific)
    parser_detect.add_argument('--config', type=str, help='Path to a detection-specific YAML configuration file.')
    # Required arguments
    parser_detect.add_argument('--date', type=str, required=True, help='Date for detection data (YYYY-MM-DD).')
    parser_detect.add_argument('--threshold', type=float, required=True, help='Anomaly score threshold.')
    # Paths (overrides defaults or config)
    parser_detect.add_argument('--model', type=str, help='Path to the trained model file (.pt).')
    parser_detect.add_argument('--scaler', type=str, help='Path to the saved scaler file (.pkl).')
    parser_detect.add_argument('--config-save-path', type=str, help='Path to the saved training config YAML (for model params).')
    # Data time range overrides (optional)
    parser_detect.add_argument('--start-time', type=str, help='Start time (HH:MM:SS). Overrides config.')
    parser_detect.add_argument('--end-time', type=str, help='End time (HH:MM:SS). Overrides config.')
    # Model/Detection parameter overrides (optional)
    parser_detect.add_argument('--window-size', type=int, help='Sliding window size (must match training). Overrides config.')
    parser_detect.add_argument('--gamma', type=float, help='Weight for reconstruction error in anomaly score. Overrides config.')
    parser_detect.add_argument('--use-cuda', action=argparse.BooleanOptionalAction, help='Enable/disable CUDA usage. Overrides config.')
    # Add argument for correction later if needed
    # parser_detect.add_argument('--apply-correction', action='store_true', help='Attempt to apply correction measures (placeholder).')
    parser_detect.set_defaults(func=detect_anomalies)

    args = parser.parse_args()

    logger.info(f"Running in '{args.mode}' mode.")
    # Call the function associated with the chosen mode
    if hasattr(args, 'func'):
        args.func(args)
    else:
        # This should not happen if subparsers are required
        logger.error("No mode selected or mode function not found.")
        parser.print_help()

if __name__ == "__main__":
    main()
