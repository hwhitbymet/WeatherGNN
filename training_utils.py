import jax
import jax.numpy as jnp
import optax
import dask.array as da
import haiku as hk
from typing import Dict, Tuple, Callable, Optional
from dataclasses import dataclass

@dataclass
class TrainState:
    """Container for training state"""
    step: int
    params: hk.Params
    opt_state: optax.OptState
    rng: jax.random.PRNGKey
    best_val_loss: float = float('inf')
    best_params: Optional[hk.Params] = None

def create_loss_fn(forward_fn: Callable) -> Callable:
    """
    Creates the loss function for the weather prediction model.
    
    Args:
        forward_fn: The model's forward pass function
        
    Returns:
        Loss function that takes parameters and batch data
    """
    def loss_fn(params: hk.Params,
                rng: jax.random.PRNGKey,
                current_data: Dict[str, jnp.ndarray],
                next_data: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, Dict]:
        """
        Compute MSE loss between model prediction and next timestep.
        """
        # Get model prediction
        predicted = forward_fn.apply(params, rng, current_data)
        
        # Combine next timestep variables into single array matching prediction shape
        actual = jnp.concatenate([
            next_data[var].reshape(-1, predicted.nodes.shape[-1] // len(next_data))
            for var in sorted(next_data.keys())
        ], axis=1)
        
        # Compute MSE loss
        mse_loss = jnp.mean((predicted.nodes - actual) ** 2)
        
        # Compute additional metrics
        metrics = {
            'mse': mse_loss,
            'rmse': jnp.sqrt(mse_loss),
            'mae': jnp.mean(jnp.abs(predicted.nodes - actual))
        }
        
        return mse_loss, metrics
    
    return loss_fn

def create_eval_step(forward_fn: Callable) -> Callable:
    """Creates evaluation step function"""
    loss_fn = create_loss_fn(forward_fn)
    
    def eval_step(params: hk.Params,
                 rng: jax.random.PRNGKey,
                 current_data: Dict[str, jnp.ndarray],
                 next_data: Dict[str, jnp.ndarray]) -> Dict:
        """Single evaluation step"""
        loss, metrics = loss_fn(params, rng, current_data, next_data)
        return metrics
    
    return jax.jit(eval_step)

def create_train_step(forward_fn: Callable,
                     optimizer: optax.GradientTransformation) -> Callable:
    """Creates training step function"""
    loss_fn = create_loss_fn(forward_fn)
    
    def train_step(state: TrainState,
                  current_data: Dict[str, jnp.ndarray],
                  next_data: Dict[str, jnp.ndarray]) -> Tuple[TrainState, Dict]:
        """Single training step"""
        rng, step_rng = jax.random.split(state.rng)
        
        # Get loss value and gradients
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(state.params, step_rng, current_data, next_data)
        
        # Update parameters
        updates, new_opt_state = optimizer.update(grads, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        
        # Update training state
        new_state = TrainState(
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            rng=rng,
            best_val_loss=state.best_val_loss,
            best_params=state.best_params
        )
        
        return new_state, metrics
    
    return jax.jit(train_step)

def create_train_state(params: hk.Params,
                      optimizer: optax.GradientTransformation,
                      rng: jax.random.PRNGKey) -> TrainState:
    """Initialize training state"""
    return TrainState(
        step=0,
        params=params,
        opt_state=optimizer.init(params),
        rng=rng,
        best_val_loss=float('inf'),
        best_params=None
    )

def evaluate_split(eval_step_fn: Callable,
                  params: hk.Params,
                  rng: jax.random.PRNGKey,
                  data: Dict[str, da.Array],
                  batch_size: int) -> Dict[str, float]:
    """Evaluate model on a data split"""
    num_samples = len(next(iter(data.values())))
    steps = num_samples // batch_size
    metrics_list = []
    
    for step in range(steps):
        # Get batch indices
        start_idx = step * batch_size
        end_idx = start_idx + batch_size
        
        # Get current and next timestep data
        current_batch = {
            var: data[var][start_idx:end_idx].compute() 
            for var in data.keys()
        }
        next_batch = {
            var: data[var][start_idx+1:end_idx+1].compute() 
            for var in data.keys()
        }
        
        # Compute metrics
        batch_metrics = eval_step_fn(params, rng, current_batch, next_batch)
        metrics_list.append(batch_metrics)
    
    # Average metrics across batches
    avg_metrics = {
        k: float(jnp.mean([m[k] for m in metrics_list]))
        for k in metrics_list[0].keys()
    }
    
    return avg_metrics

def training_loop(state: TrainState,
                 train_data: Dict[str, da.Array],
                 val_data: Dict[str, da.Array],
                 batch_size: int,
                 num_epochs: int,
                 train_step_fn: Callable,
                 eval_step_fn: Callable,
                 early_stopping_patience: Optional[int] = None) -> TrainState:
    """
    Main training loop with validation
    
    Args:
        state: Initial training state
        train_data: Training data dictionary of Dask arrays
        val_data: Validation data dictionary of Dask arrays
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        train_step_fn: Compiled training step function
        eval_step_fn: Compiled evaluation step function
        early_stopping_patience: Optional patience for early stopping
        
    Returns:
        Final training state
    """
    num_samples = len(next(iter(train_data.values())))
    steps_per_epoch = num_samples // batch_size
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        epoch_metrics = []
        for step in range(steps_per_epoch):
            # Get batch indices
            start_idx = step * batch_size
            end_idx = start_idx + batch_size
            
            # Get current and next timestep data
            current_batch = {
                var: train_data[var][start_idx:end_idx].compute() 
                for var in train_data.keys()
            }
            next_batch = {
                var: train_data[var][start_idx+1:end_idx+1].compute() 
                for var in train_data.keys()
            }
            
            # Perform training step
            state, metrics = train_step_fn(state, current_batch, next_batch)
            epoch_metrics.append(metrics)
        
        # Compute training metrics
        train_metrics = {
            k: float(jnp.mean([m[k] for m in epoch_metrics]))
            for k in epoch_metrics[0].keys()
        }
        
        # Validation step
        val_metrics = evaluate_split(
            eval_step_fn, state.params, state.rng, val_data, batch_size)
        
        # Update best parameters if validation loss improved
        if val_metrics['mse'] < state.best_val_loss:
            state = state._replace(
                best_val_loss=val_metrics['mse'],
                best_params=state.params
            )
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print metrics
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Training metrics:")
        for k, v in train_metrics.items():
            print(f"  {k}: {v:.6f}")
        print(f"Validation metrics:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.6f}")
        
        # Early stopping check
        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Restore best parameters
    if state.best_params is not None:
        state = state._replace(params=state.best_params)
    
    return state