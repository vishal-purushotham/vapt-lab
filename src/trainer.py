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

    def _get_target_output(self, data: torch.Tensor) -> torch.Tensor:
        """Extracts the target dimensions from data if target_dims is specified."""
        if self.target_dims is None:
            return data
        elif isinstance(self.target_dims, int):
            return data[:, :, [self.target_dims]] # Ensure it remains 3D if single dim
        else:
            return data[:, :, self.target_dims]

    def _process_batch(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        """Processes a single batch for loss calculation."""
        x = x.to(self.device)
        y = y.to(self.device)

        preds, recons = self.model(x)

        # Select target dimensions if specified
        x_target = self._get_target_output(x)
        y_target = self._get_target_output(y)

        # Adjust dimensions for loss calculation if necessary (e.g., remove feature dim if 1)
        if y_target.shape[-1] == 1:
             y_target = y_target.squeeze(-1)
        if preds.shape[-1] == 1:
             preds = preds.squeeze(-1)
        # Similar adjustments might be needed for recons and x_target depending on model output

        # Calculate losses - Use sqrt for RMSE
        forecast_loss = torch.sqrt(self.forecast_criterion(y_target, preds))
        recon_loss = torch.sqrt(self.recon_criterion(x_target, recons))

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
