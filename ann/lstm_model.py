"""
LSTM Model Architecture for Exoplanet Classification
Defines the neural network architecture for binary classification of Kepler light curves.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
from typing import Any, Tuple, Dict, Optional
import logging
import io
import sys


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExoplanetLSTM:
    """
    LSTM model for binary classification of exoplanet light curves.
    
    Architecture:
    - Input: (sequence_length, n_features)
    - LSTM layer(s) with dropout
    - Dense layer(s) with activation
    - Output: Single sigmoid unit for binary classification
    """
    
    def __init__(self, sequence_length: int = 1000, n_features: int = 2, 
                 lstm_units: int = 64, dropout_rate: float = 0.3,
                 dense_units: int = 32, learning_rate: float = 0.001):
        """
        Initialize LSTM model parameters.
        
        Args:
            sequence_length: Length of input sequences (L)
            n_features: Number of features per timestep (flux + flux_err = 2)
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            dense_units: Number of units in dense layer
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        
        self.model = None
        self.history = None
        
    def build_model(self, use_bidirectional: bool = False, 
                   use_multiple_lstm: bool = False) -> keras.Model:
        """
        Build the LSTM model architecture.
        
        Args:
            use_bidirectional: Whether to use bidirectional LSTM
            use_multiple_lstm: Whether to use multiple LSTM layers
            
        Returns:
            Compiled Keras model
        """
        model: Any = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(self.sequence_length, self.n_features)))
        
        if use_multiple_lstm:
            # Multiple LSTM layers
            if use_bidirectional:
                model.add(layers.Bidirectional(
                    layers.LSTM(self.lstm_units, return_sequences=True),
                    name='bidirectional_lstm_1'
                ))
            else:
                model.add(layers.LSTM(self.lstm_units, return_sequences=True, name='lstm_1'))
            
            model.add(layers.Dropout(self.dropout_rate, name='dropout_1'))
            
            # Second LSTM layer
            if use_bidirectional:
                model.add(layers.Bidirectional(
                    layers.LSTM(self.lstm_units // 2),
                    name='bidirectional_lstm_2'
                ))
            else:
                model.add(layers.LSTM(self.lstm_units // 2, name='lstm_2'))
        else:
            # Single LSTM layer
            if use_bidirectional:
                model.add(layers.Bidirectional(
                    layers.LSTM(self.lstm_units),
                    name='bidirectional_lstm'
                ))
            else:
                model.add(layers.LSTM(self.lstm_units, name='lstm'))
        
        # Dropout for regularization
        model.add(layers.Dropout(self.dropout_rate, name='dropout_main'))
        
        # Dense layer
        model.add(layers.Dense(self.dense_units, activation='relu', name='dense'))
        
        # Additional dropout
        model.add(layers.Dropout(self.dropout_rate / 2, name='dropout_dense'))
        
        # Output layer for binary classification
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy', # mse
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        logger.info(f"Model built successfully. Total parameters: {model.count_params()}")
        
        return model
    
    def get_model_summary(self) -> str:
        """Get string representation of model architecture."""
        if self.model is None:
            return "Model not built yet. Call build_model() first."
        

        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        output = buffer.getvalue()
        sys.stdout = old_stdout
        
        return output
    
    def create_callbacks(self, patience: int = 10, min_delta: float = 0.001,
                        save_best_weights: bool = True, 
                        reduce_lr_patience: int = 5) -> list:
        """
        Create training callbacks for better training control.
        
        Args:
            patience: Patience for early stopping
            min_delta: Minimum change to qualify as improvement
            save_best_weights: Whether to save best weights during training
            reduce_lr_patience: Patience for learning rate reduction
            
        Returns:
            List of Keras callbacks
        """
        callback_list = []
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=save_best_weights,
            verbose=1
        )
        callback_list.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # Model checkpoint (optional)
        if save_best_weights:
            checkpoint = callbacks.ModelCheckpoint(
                filepath='best_exoplanet_model.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
            callback_list.append(checkpoint)
        
        return callback_list
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              use_callbacks: bool = True, verbose: int = 1) -> keras.callbacks.History:
        """
        Train the LSTM model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
            use_callbacks: Whether to use training callbacks
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Prepare callbacks
        callback_list = self.create_callbacks() if use_callbacks else None
        
        # Calculate class weights for imbalanced data
        pos_weight = len(y_train) / (2 * np.sum(y_train))
        neg_weight = len(y_train) / (2 * (len(y_train) - np.sum(y_train)))
        class_weight = {0: neg_weight, 1: pos_weight}
        
        logger.info(f"Training with class weights: {class_weight}")
        logger.info(msg=f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callback_list,
            verbose=verbose
        )
        
        logger.info("Training completed!")
        return self.history
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input data
            batch_size: Batch size for prediction
            
        Returns:
            Predictions (probabilities)
        """
        if self.model is None:
            raise ValueError("Model not built or trained.")
        
        predictions = self.model.predict(X, batch_size=batch_size)
        return predictions.flatten()
    
    def predict_classes(self, X: np.ndarray, threshold: float = 0.5,
                       batch_size: int = 32) -> np.ndarray:
        """
        Predict binary classes.
        
        Args:
            X: Input data
            threshold: Classification threshold
            batch_size: Batch size for prediction
            
        Returns:
            Binary predictions (0 or 1)
        """
        probabilities = self.predict(X, batch_size)
        return (probabilities >= threshold).astype(int)
    
    def save_model(self, filepath: str, save_weights_only: bool = False):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        if save_weights_only:
            self.model.save_weights(filepath)
        else:
            self.model.save(filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, load_weights_only: bool = False):
        """Load a saved model."""
        if load_weights_only:
            if self.model is None:
                raise ValueError("Model architecture must be built before loading weights.")
            self.model.load_weights(filepath)
        else:
            self.model = keras.models.load_model(filepath)
            
        logger.info(f"Model loaded from {filepath}")
    
    def get_training_plots_data(self) -> Dict:
        """
        Get training history data for plotting.
        
        Returns:
            Dictionary with training metrics
        """
        if self.history is None:
            return {}
        
        return {
            'epochs': list(range(1, len(self.history.history['loss']) + 1)),
            'train_loss': self.history.history['loss'],
            'val_loss': self.history.history['val_loss'],
            'train_accuracy': self.history.history['accuracy'],
            'val_accuracy': self.history.history['val_accuracy'],
            'train_precision': self.history.history.get('precision', []),
            'val_precision': self.history.history.get('val_precision', []),
            'train_recall': self.history.history.get('recall', []),
            'val_recall': self.history.history.get('val_recall', [])
        }


def create_advanced_model(sequence_length: int = 1000, n_features: int = 2) -> ExoplanetLSTM:
    """
    Create an advanced LSTM model with bidirectional layers and attention.
    
    Args:
        sequence_length: Length of input sequences
        n_features: Number of features per timestep
        
    Returns:
        ExoplanetLSTM instance with advanced architecture
    """
    model = ExoplanetLSTM(
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_units=128,
        dropout_rate=0.3,
        dense_units=64,
        learning_rate=0.001
    )
    
    # Build with bidirectional and multiple LSTM layers
    model.build_model(use_bidirectional=True, use_multiple_lstm=True)
    
    return model


def create_simple_model(sequence_length: int = 1000, n_features: int = 2) -> ExoplanetLSTM:
    """
    Create a simple LSTM model for quick testing.
    
    Args:
        sequence_length: Length of input sequences
        n_features: Number of features per timestep
        
    Returns:
        ExoplanetLSTM instance with simple architecture
    """
    model = ExoplanetLSTM(
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_units=64,
        dropout_rate=0.3,
        dense_units=32,
        learning_rate=0.001
    )
    
    # Build simple model
    model.build_model(use_bidirectional=False, use_multiple_lstm=False)
    
    return model


if __name__ == "__main__":
    # Example usage and testing
    
    # Create simple model
    model = create_simple_model()
    print("Simple Model Architecture:")
    print(model.get_model_summary())
    
    # Create advanced model
    advanced_model = create_advanced_model()
    print("\nAdvanced Model Architecture:")
    print(advanced_model.get_model_summary())
    
    # Test with dummy data
    batch_size = 16
    sequence_length = 1000
    n_features = 2
    
    X_dummy = np.random.randn(batch_size, sequence_length, n_features)
    y_dummy = np.random.randint(0, 2, batch_size)
    
    print(f"\nTesting prediction with dummy data:")
    print(f"Input shape: {X_dummy.shape}")
    
    predictions = model.predict(X_dummy)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    
    class_predictions = model.predict_classes(X_dummy)
    print(f"Class predictions: {class_predictions[:5]}")