import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer class for the MTAD-GAT model."""
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        window_size: int,
        n_features: int,
        n_epochs: int,
        batch_size: int,
        target_dims: list = None, # Indices of features to target for forecasting/reconstruction
        init_lr: float = 0.001,
        forecast_criterion = nn.MSELoss(),
        recon_criterion = nn.MSELoss(),
        use_cuda: bool = True,
        save_path: str = "./output/models", # Path to save best model
        log_dir: str = "./output/logs",     # Path for TensorBoard logs
        print_every: int = 1,
        log_tensorboard: bool = True,
        args_summary: str = "",
    ):
        """
        Args:
            model: The PyTorch model to train (e.g., MTAD_GAT).
            optimizer: The optimizer to use (e.g., Adam).
            window_size: Length of the input sequence window.
            n_features: Number of features in the input data.
            n_epochs: Number of training epochs.
            batch_size: Number of samples per batch.
            target_dims: List of feature indices to focus on for loss calculation. If None, uses all features.
            init_lr: Initial learning rate.
            forecast_criterion: Loss function for forecasting task.
            recon_criterion: Loss function for reconstruction task.
            use_cuda: Whether to use GPU if available.
            save_path: Directory to save the best trained model.
            log_dir: Directory for TensorBoard logging.
            print_every: Print training progress every X epochs.
            log_tensorboard: Whether to log losses to TensorBoard.
            args_summary: A string summary of arguments/config to log.
        """
        self.model = model
        self.optimizer = optimizer
        self.window_size = window_size
        self.n_features = n_features
        self.target_dims = target_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.forecast_criterion = forecast_criterion
        self.recon_criterion = recon_criterion
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.save_path = save_path
        self.log_dir = log_dir
        self.print_every = print_every
        self.log_tensorboard = log_tensorboard
        self.best_val_loss = float('inf')

        self.losses = {
            "train_total": [], "train_forecast": [], "train_recon": [],
            "val_total": [], "val_forecast": [], "val_recon": [],
        }
        self.epoch_times = []

        if self.device == "cuda":
            self.model.cuda()
            logger.info("Training on GPU.")
        else:
            logger.info("Training on CPU.")

        # Create directories if they don't exist
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        if self.log_tensorboard:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            if args_summary:
                self.writer.add_text("args_summary", args_summary)
            logger.info(f"TensorBoard logging enabled. Log directory: {self.log_dir}")

    def _build_model(self, **params):
        """Builds the MTAD-GAT model."""
        logger = logging.getLogger(__name__)

        try:
            # Extract parameters, providing defaults
            n_features = params['n_features']
            window_size = params['window_size']
            hidden_size = params.get('hidden_size', 150) # Used for gru_hid_dim
            # No longer using fc_dropout, gru_dropout, target_dims directly from top-level config here
            gru_layers = params.get('gru_layers', 1)
            use_gatv2 = params.get('use_gatv2', True)
            use_bias = params.get('use_bias', True)
            dropout = params.get('dropout', 0.3) # Get the single dropout value

            model = MTAD_GAT(
                n_features=n_features,
                window_size=window_size,
                out_dim=n_features, # Predict all features
                gru_hid_dim=hidden_size, # Pass GRU hidden size
                gru_n_layers=gru_layers,
                dropout=dropout, # Pass the single dropout value
                use_gatv2=use_gatv2,
                # Pass other MTAD_GAT specific params, getting from config or using defaults
                kernel_size=params.get('kernel_size', 7),
                feat_gat_embed_dim=params.get('feat_gat_embed_dim', None),
                time_gat_embed_dim=params.get('time_gat_embed_dim', None),
                forecast_n_layers=params.get('forecast_n_layers', 1),
                forecast_hid_dim=params.get('forecast_hid_dim', 150),
                recon_n_layers=params.get('recon_n_layers', 1),
                recon_hid_dim=params.get('recon_hid_dim', 150),
                alpha=params.get('alpha', 0.2)
                # Note: MTAD_GAT doesn't seem to use use_bias directly in its __init__ signature
            )
            logger.info(f"MTAD-GAT model built successfully.")
            return model
        except KeyError as e:
            logger.error(f"Missing required parameter for model building: {e}")

    def _get_target_output(self, data: torch.Tensor) -> torch.Tensor:
        """Selects the target dimensions from the data based on self.target_dims.

        Args:
            data (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Target data tensor with selected dimensions.
        """
        if self.target_dims is None:
            return data

        # Ensure target_dims is a list
        dims_to_select = self.target_dims
        if isinstance(dims_to_select, int):
            dims_to_select = [dims_to_select]

        if data.ndim == 3: # Input like x (batch, window, features)
            target_data = data[:, :, dims_to_select]
        elif data.ndim == 2: # Input like y or x[:,-1,:] (batch, features)
            target_data = data[:, dims_to_select]
        else:
            raise ValueError(f"_get_target_output received tensor with unexpected ndim: {data.ndim}")

        return target_data

    def _process_batch(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        """Process a single batch of data for training.

        Args:
            x (torch.Tensor): Input data tensor (batch_size, window_size, n_features).
            y (torch.Tensor): Target data tensor (batch_size, forecast_horizon, n_features).

        Returns:
            tuple: forecast_loss, recon_loss, total_loss.
        """
        x = x.to(self.device)
        y = y.to(self.device)

        preds, recons = self.model(x)

        # --- Match original idps-escape logic ---
        if self.target_dims is not None:
            x = x[:, :, self.target_dims]
            y = y[:, :, self.target_dims].squeeze(-1) # Filter and squeeze y's last dim
            # Filter recons to match filtered x if target_dims is used
            recons = recons[:, :, self.target_dims]

        # --- Debug: Log shapes before loss calculation ---
        logger.debug(f"Shape pre-loss - y: {y.shape}, preds: {preds.shape}")
        logger.debug(f"Shape pre-loss - x: {x.shape}, recons: {recons.shape}")

        # Squeeze dimension 1 if necessary (original logic)
        if preds.ndim == 3:
            preds = preds.squeeze(1) # Squeeze the forecast horizon dimension if present
        if y.ndim == 3:
            # Only squeeze if the dimension is indeed 1 (e.g. forecast horizon=1)
            if y.shape[1] == 1:
                y = y.squeeze(1) # Squeeze dim 1 of y
            # If forecast horizon > 1, ensure preds also has that dimension or handle mismatch
            # Current MTAD_GAT returns (batch, features) for preds, need alignment if y_target is (batch, horizon>1, features)
            # For now, assume horizon=1 or model/target mismatch needs addressing elsewhere
            elif preds.ndim == 2: # preds is (batch, features), y is (batch, horizon>1, features)
                logger.warning(f"Potential forecast shape mismatch: preds {preds.shape} vs y {y.shape}. Using y[:, -1, :] for loss.")
                y = y[:, -1, :] # Use last step of target horizon for now

        # Calculate losses
        forecast_loss = torch.sqrt(self.forecast_criterion(y, preds))
        # Compare recons (batch, features) with the LAST step of x (batch, features)
        x_last_step = x[:, -1, :]
        recon_loss = torch.sqrt(self.recon_criterion(x_last_step, recons))

        # Debug shapes *after* potential filtering/squeezing
        logger.debug(f"Shape final - y: {y.shape}, preds: {preds.shape}")
        logger.debug(f"Shape final - x: {x.shape}, recons: {recons.shape}")

        total_loss = forecast_loss + recon_loss
        return forecast_loss, recon_loss, total_loss

    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Trains the model using the provided data loaders."""
        logger.info(f"Starting training for {self.n_epochs} epochs...")
        train_start = time.time()

        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            self.model.train()
            epoch_train_forecast_losses = []
            epoch_train_recon_losses = []

            # --- Training Loop ---
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
 
                preds, recons = self.model(x)

                self.optimizer.zero_grad()
                forecast_loss, recon_loss, total_loss = self._process_batch(x, y)
                total_loss.backward()
                self.optimizer.step()

                epoch_train_forecast_losses.append(forecast_loss.item()**2) # Store squared errors for correct RMSE calc
                epoch_train_recon_losses.append(recon_loss.item()**2)

            # Calculate average RMSE for the epoch
            train_forecast_epoch_loss = np.sqrt(np.mean(epoch_train_forecast_losses))
            train_recon_epoch_loss = np.sqrt(np.mean(epoch_train_recon_losses))
            train_total_epoch_loss = train_forecast_epoch_loss + train_recon_epoch_loss

            self.losses["train_forecast"].append(train_forecast_epoch_loss)
            self.losses["train_recon"].append(train_recon_epoch_loss)
            self.losses["train_total"].append(train_total_epoch_loss)

            # --- Validation Loop ---
            val_forecast_epoch_loss, val_recon_epoch_loss, val_total_epoch_loss = float('nan'), float('nan'), float('nan')
            if val_loader is not None:
                val_forecast_epoch_loss, val_recon_epoch_loss, val_total_epoch_loss = self.evaluate(val_loader)
                self.losses["val_forecast"].append(val_forecast_epoch_loss)
                self.losses["val_recon"].append(val_recon_epoch_loss)
                self.losses["val_total"].append(val_total_epoch_loss)

                # Save model if validation loss improves
                if val_total_epoch_loss < self.best_val_loss:
                    self.best_val_loss = val_total_epoch_loss
                    self.save()
                    logger.info(f"Epoch {epoch+1}: New best model saved with val_loss={self.best_val_loss:.5f}")
            else:
                # If no validation, the model is saved at the end of training (see below)
                pass # Explicitly do nothing here, save happens after loop

            # --- Logging ---
            if self.log_tensorboard:
                self.writer.add_scalar('Loss/Train_Forecast', train_forecast_epoch_loss, epoch)
                self.writer.add_scalar('Loss/Train_Recon', train_recon_epoch_loss, epoch)
                self.writer.add_scalar('Loss/Train_Total', train_total_epoch_loss, epoch)
                if val_loader is not None:
                    self.writer.add_scalar('Loss/Val_Forecast', val_forecast_epoch_loss, epoch)
                    self.writer.add_scalar('Loss/Val_Recon', val_recon_epoch_loss, epoch)
                    self.writer.add_scalar('Loss/Val_Total', val_total_epoch_loss, epoch)
                self.writer.flush()

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            if (epoch + 1) % self.print_every == 0:
                log_msg = (
                    f"[Epoch {epoch + 1}/{self.n_epochs}] "
                    f"Tr_F: {train_forecast_epoch_loss:.5f}, Tr_R: {train_recon_epoch_loss:.5f}, Tr_T: {train_total_epoch_loss:.5f}"
                )
                if val_loader is not None:
                    log_msg += (
                        f" | Val_F: {val_forecast_epoch_loss:.5f}, Val_R: {val_recon_epoch_loss:.5f}, Val_T: {val_total_epoch_loss:.5f}"
                    )
                log_msg += f" [{epoch_time:.1f}s]"
                logger.info(log_msg)

        # Save the model at the end of training if no validation set was used
        if val_loader is None:
            logger.info("Validation skipped. Saving final model.")
            self.save(final=True)

        train_time = int(time.time() - train_start)
        logger.info(f"Training finished in {train_time}s.")
        if self.log_tensorboard:
            self.writer.add_text("total_train_time", str(train_time))
            self.writer.close()

    def evaluate(self, data_loader: DataLoader) -> tuple:
        """Evaluates the model on the given data loader."""
        self.model.eval()
        epoch_forecast_losses = []
        epoch_recon_losses = []

        with torch.no_grad():
            for x, y in data_loader:
                forecast_loss, recon_loss, _ = self._process_batch(x, y)
                epoch_forecast_losses.append(forecast_loss.item()**2)
                epoch_recon_losses.append(recon_loss.item()**2)

        forecast_epoch_loss = np.sqrt(np.mean(epoch_forecast_losses))
        recon_epoch_loss = np.sqrt(np.mean(epoch_recon_losses))
        total_epoch_loss = forecast_epoch_loss + recon_epoch_loss

        return forecast_epoch_loss, recon_epoch_loss, total_epoch_loss

    def save(self, final: bool = False):
        """Saves the model state dictionary."""
        model_filename = 'model_final.pt' if final else 'model_best.pt'
        save_file = os.path.join(self.save_path, model_filename)
        torch.save(self.model.state_dict(), save_file)
        if not final:
             logger.debug(f"Saved best model state to {save_file}")
        else:
             logger.info(f"Saved final model state to {save_file}")
