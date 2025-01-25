import logging
import time
import jax
import jax.numpy as jnp
import jraph
import optax
import os
import haiku as hk
import pickle
import argparse
from tqdm import tqdm

from load_data import load_netcdf_to_zarr, get_data_splits
from config import Configuration
from weather_gnn import WeatherPrediction

def create_forward_fn(config, rng_key):
    """Create the forward pass function with Haiku transform"""
    def forward_fn(latlon_data: dict[str, jnp.ndarray]) -> jnp.ndarray:
        model = WeatherPrediction(config, rng_key)
        return model(latlon_data)
    
    return hk.transform(forward_fn)

def train(
    config: Configuration, 
    model: hk.TransformedWithState,
    params: dict, 
    splits: dict
):
    """
    Train the weather prediction model with validation and detailed logging
    
    Args:
        config: Configuration object
        model: Haiku transformed model
        params: Initial model parameters
        splits: Data splits dictionary
    
    Returns:
        Trained model parameters
    """
    # Setup optimizer
    optimizer = optax.adam(learning_rate=1e-3)
    train_step = create_train_step(model, optimizer)
    
    # Random number generator
    rng = jax.random.PRNGKey(42)
    
    # Extract data splits
    train_data = splits['train']
    val_data = splits['validation']
    
    # Determine number of timesteps (assumes first variable's data represents all)
    first_var = next(iter(train_data.keys()))
    train_timesteps = train_data[first_var].shape[0]
    val_timesteps = val_data[first_var].shape[0]
    
    logging.info(f"Training data timesteps: {train_timesteps}")
    logging.info(f"Validation data timesteps: {val_timesteps}")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Outer epoch progress bar
    epoch_progress = tqdm(
        range(config.training.num_epochs), 
        desc="Epochs", 
        position=0, 
        leave=True
    )
    
    for epoch in epoch_progress:
        # Training phase
        train_loss = 0.0
        
        # Training timestep progress bar
        train_timestep_progress = tqdm(
            range(train_timesteps - 1), 
            desc=f"Training Epoch {epoch+1}", 
            position=1, 
            leave=False
        )
        
        for t in train_timestep_progress:
            # Prepare input (current timestep) and target (next timestep)
            input_data = {var: train_data[var][t] for var in train_data}
            target_data = {var: train_data[var][t+1] for var in train_data}
            
            # Separate random key for this step
            step_rng = jax.random.fold_in(rng, t)
            
            # Perform training step
            params, loss = train_step(params, step_rng, input_data, target_data)
            train_loss += loss
            
            # Update training timestep progress bar
            train_timestep_progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation phase
        val_loss = 0.0
        
        # Validation timestep progress bar
        val_timestep_progress = tqdm(
            range(val_timesteps - 1), 
            desc=f"Validation Epoch {epoch+1}", 
            position=1, 
            leave=False
        )
        
        for t in val_timestep_progress:
            # Prepare input (current timestep) and target (next timestep)
            input_data = {var: val_data[var][t] for var in val_data}
            target_data = {var: val_data[var][t+1] for var in val_data}
            
            # Compute validation loss (without gradient updates)
            pred = model.apply(params, rng, input_data)
            loss = compute_loss(pred, target_data)
            val_loss += loss
            
            # Update validation timestep progress bar
            val_timestep_progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Normalize losses
        train_loss /= (train_timesteps - 1)
        val_loss /= (val_timesteps - 1)
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Update epoch progress bar
        epoch_progress.set_postfix({
            'Train Loss': f'{train_loss.item():.4f}', 
            'Val Loss': f'{val_loss.item():.4f}', 
            'Patience': patience_counter
        })
        
        # Detailed logging
        logging.info(f"Epoch {epoch+1}/{config.training.num_epochs}")
        logging.info(f"  Train Loss: {train_loss:.4f}")
        logging.info(f"  Validation Loss: {val_loss:.4f}")
        logging.info(f"  Patience Counter: {patience_counter}")
        
        # Early stopping
        if patience_counter >= config.training.early_stopping_patience:
            logging.info("Early stopping triggered")
            break
    
    return params

def create_train_step(forward_fn, optimizer):
    """
    Create a single training step function
    
    Args:
        forward_fn: Haiku transformed forward function
        optimizer: Optax optimizer
    
    Returns:
        Jitted training step function
    """
    def train_step(params, rng, input_data, target_data):
        """
        Single training step
        
        Args:
            params: Model parameters
            rng: Random number generator
            input_data: Input data for current timestep
            target_data: Ground truth for next timestep
        
        Returns:
            Tuple of (updated params, loss)
        """
        def loss_fn(params):
            pred = forward_fn.apply(params, rng, input_data)
            return compute_loss(pred, target_data)
        
        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(params)
        
        # Update parameters
        updates, _ = optimizer.update(grads, None)
        updated_params = optax.apply_updates(params, updates)
        
        return updated_params, loss
    
    return jax.jit(train_step)

def compute_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Mean Squared Error loss between prediction and ground truth
    
    Args:
        pred: Model prediction of shape (n_nodes, n_features)
        target: Ground truth of shape (n_nodes, n_features)
    
    Returns:
        Scalar loss value
    """
    return jnp.mean(jnp.square(pred - target))

def main(config_path: str):
    # Setup logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    # Log start of training
    logging.info("Starting Weather Prediction Model Training")
    logging.info(f"Configuration file: {config_path}")
    
    # Load configuration
    config = Configuration.load(config_path)
    
    # Get data splits
    logging.info("Loading data splits...")
    splits = get_data_splits(config)
    logging.info("Data splits loaded successfully")
    
    # Initialize model
    rng = jax.random.PRNGKey(42)
    logging.info("Creating model forward function...")
    model = create_forward_fn(config.model, rng)

    # Initialize parameters
    logging.info("Initializing model parameters...")
    if not os.path.exists(config.data.init_params_cache):
        init_data = {var: splits['train'][var][0][:] 
                     for var in splits['train'].keys()}
        params = model.init(rng, init_data, config.model)
        with open(config.data.init_params_cache, 'wb') as f:
            pickle.dump(params, f)
        logging.info(f"Saved initial parameters to {config.data.init_params_cache}")
    else:
        with open(config.data.init_params_cache, 'rb') as f:
            params = pickle.load(f)
        logging.info("Loaded cached initial parameters")
    
    # Train the model
    logging.info("Starting model training...")
    trained_params = train(
        config, 
        model, 
        params, 
        splits, 
    )
    
    # Optional: Save trained parameters
    trained_params_path = os.path.join(log_dir, f"trained_params_{time.strftime('%Y%m%d_%H%M%S')}.pkl")
    with open(trained_params_path, 'wb') as f:
        pickle.dump(trained_params, f)
    
    logging.info(f"Training complete. Trained parameters saved to {trained_params_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/prototype.yaml')
    parser.add_argument('--log-level', default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    main(args.config)