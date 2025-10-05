"""
Data Preprocessing for LSTM Exoplanet Classification
Handles data cleaning, normalization, and sequence generation for time series data.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightCurvePreprocessor:
    """
    Preprocesses Kepler light curve data for LSTM training.
    
    Main functions:
    1. Data cleaning (quality filtering, NaN removal)
    2. Flux normalization
    3. Sequence generation with fixed length
    4. Train/validation/test splitting
    """
    
    def __init__(self, sequence_length: int = 1000, stride_ratio: float = 0.5, 
                 min_length_ratio: float = 0.8):
        """
        Initialize preprocessor with sequence parameters.
        
        Args:
            sequence_length: Fixed length for LSTM sequences (L)
            stride_ratio: Stride as ratio of sequence_length for overlapping sequences
            min_length_ratio: Minimum length ratio to keep a light curve
        """
        self.sequence_length = sequence_length
        self.stride = int(sequence_length * stride_ratio)
        self.min_length = int(sequence_length * min_length_ratio)
        
    def clean_light_curve(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean a single light curve by removing bad quality data and NaN values.
        
        Args:
            df: DataFrame with columns [time, pdcsap_flux, pdcsap_flux_err, quality]
            
        Returns:
            Cleaned DataFrame
        """
        original_length = len(df)
        
        # Convert data types to handle big-endian issues
        df = df.copy()
        for col in ['time', 'pdcsap_flux', 'pdcsap_flux_err', 'quality']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove bad quality data (quality > 0)
        df_clean = df[df['quality'] == 0].copy()
        
        # Remove NaN and Inf values
        df_clean = df_clean.dropna(subset=['pdcsap_flux', 'pdcsap_flux_err'])
        df_clean = df_clean[np.isfinite(df_clean['pdcsap_flux'])]
        df_clean = df_clean[np.isfinite(df_clean['pdcsap_flux_err'])]
        
        # Remove outliers (beyond 5 sigma)
        if len(df_clean) > 0:
            flux_median = df_clean['pdcsap_flux'].median()
            flux_std = df_clean['pdcsap_flux'].std()
            
            if flux_std > 0:
                outlier_mask = np.abs(df_clean['pdcsap_flux'] - flux_median) < 5 * flux_std
                df_clean = df_clean[outlier_mask]
        
        logger.debug(f"Cleaned light curve: {original_length} -> {len(df_clean)} points")
        return df_clean.reset_index(drop=True)
    
    def normalize_flux(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize flux using z-score normalization for each light curve.
        
        Formula: Flux_norm = (Flux - median(Flux)) / std(Flux)
        """
        df_norm = df.copy()
        
        flux_median = df['pdcsap_flux'].median()
        flux_std = df['pdcsap_flux'].std()
        
        if flux_std > 0:
            df_norm['pdcsap_flux_norm'] = (df['pdcsap_flux'] - flux_median) / flux_std
            # Also normalize the error
            df_norm['pdcsap_flux_err_norm'] = df['pdcsap_flux_err'] / flux_std
        else:
            # Handle constant flux case
            df_norm['pdcsap_flux_norm'] = 0.0
            df_norm['pdcsap_flux_err_norm'] = 0.0
            
        return df_norm
    
    def generate_sequences(self, df: pd.DataFrame, label: int, kic_id: str) -> List[Tuple[np.ndarray, int, str]]:
        """
        Generate fixed-length sequences from a light curve.
        
        Args:
            df: Preprocessed light curve DataFrame
            label: Class label (0 or 1)
            kic_id: KIC identifier
            
        Returns:
            List of (sequence_array, label, kic_id) tuples
        """
        sequences = []
        
        # Check if light curve is long enough
        if len(df) < self.min_length:
            logger.debug(f"Light curve {kic_id} too short: {len(df)} < {self.min_length}")
            return sequences
        
        # Use normalized flux and flux error as features (ensure proper data types)
        features = df[['pdcsap_flux_norm', 'pdcsap_flux_err_norm']].values.astype(np.float32)
        
        # Generate overlapping sequences
        start_idx = 0
        while start_idx + self.sequence_length <= len(features):
            sequence = features[start_idx:start_idx + self.sequence_length]
            sequences.append((sequence, label, kic_id))
            start_idx += self.stride
        
        logger.debug(f"Generated {len(sequences)} sequences from light curve {kic_id}")
        return sequences
    
    def process_single_light_curve(self, df: pd.DataFrame, label: int, kic_id: str) -> List[Tuple[np.ndarray, int, str]]:
        """
        Complete preprocessing pipeline for a single light curve.
        
        Returns:
            List of processed sequences
        """
        try:
            # Clean the data
            df_clean = self.clean_light_curve(df)
            
            if len(df_clean) < self.min_length:
                return []
            
            # Normalize flux
            df_norm = self.normalize_flux(df_clean)
            
            # Generate sequences
            sequences = self.generate_sequences(df_norm, label, kic_id)
            
            return sequences
            
        except Exception as e:
            logger.error(f"Error processing light curve {kic_id}: {e}")
            return []
    
    def process_all_data(self, data_list: List[Tuple[pd.DataFrame, int, str]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Process all light curves and return arrays suitable for LSTM training.
        
        Args:
            data_list: List of (dataframe, label, kic_id) tuples
            
        Returns:
            X: Feature array of shape (n_sequences, sequence_length, n_features)
            y: Label array of shape (n_sequences,)
            kic_ids: List of KIC IDs for each sequence
        """
        all_sequences = []
        all_labels = []
        all_kic_ids = []
        
        logger.info(f"Processing {len(data_list)} light curves...")
        
        for i, (df, label, kic_id) in enumerate(data_list):
            sequences = self.process_single_light_curve(df, label, kic_id)
            
            for seq, seq_label, seq_kic_id in sequences:
                all_sequences.append(seq)
                all_labels.append(seq_label)
                all_kic_ids.append(seq_kic_id)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(data_list)} light curves...")
        
        # Convert to numpy arrays
        X = np.array(all_sequences)
        y = np.array(all_labels)
        
        logger.info(f"Final dataset shape: X={X.shape}, y={y.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        return X, y, all_kic_ids
    
    def create_train_val_test_split(self, X: np.ndarray, y: np.ndarray, kic_ids: List[str], 
                                   train_size: float = 0.7, val_size: float = 0.15, 
                                   test_size: float = 0.15, random_state: int = 42) -> Dict:
        """
        Split data into train/validation/test sets ensuring no data leakage by KIC ID.
        
        Args:
            X: Feature array
            y: Label array
            kic_ids: List of KIC IDs
            train_size, val_size, test_size: Split ratios (should sum to 1.0)
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with train/val/test splits
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split ratios must sum to 1.0"
        
        # Get unique KIC IDs and their labels
        unique_kics = list(set(kic_ids))
        kic_labels = {}
        for kic in unique_kics:
            # Get the label for this KIC (should be consistent across sequences)
            indices = [i for i, k in enumerate(kic_ids) if k == kic]
            kic_labels[kic] = y[indices[0]]
        
        # Split by KIC IDs to prevent data leakage
        kic_labels_list = [(kic, label) for kic, label in kic_labels.items()]
        kics_only = [kic for kic, _ in kic_labels_list]
        labels_only = [label for _, label in kic_labels_list]
        
        # First split: train vs (val + test)
        train_kics, temp_kics, _, temp_labels = train_test_split(
            kics_only, labels_only, 
            train_size=train_size, 
            stratify=labels_only,
            random_state=random_state
        )
        
        # Second split: val vs test
        val_ratio = val_size / (val_size + test_size)
        val_kics, test_kics = train_test_split(
            temp_kics, 
            train_size=val_ratio,
            stratify=temp_labels,
            random_state=random_state
        )
        
        # Create index mappings
        train_indices = [i for i, kic in enumerate(kic_ids) if kic in train_kics]
        val_indices = [i for i, kic in enumerate(kic_ids) if kic in val_kics]
        test_indices = [i for i, kic in enumerate(kic_ids) if kic in test_kics]
        
        # Create splits
        splits = {
            'X_train': X[train_indices],
            'y_train': y[train_indices],
            'kic_train': [kic_ids[i] for i in train_indices],
            
            'X_val': X[val_indices],
            'y_val': y[val_indices],
            'kic_val': [kic_ids[i] for i in val_indices],
            
            'X_test': X[test_indices],
            'y_test': y[test_indices],
            'kic_test': [kic_ids[i] for i in test_indices],
            
            'train_kics': train_kics,
            'val_kics': val_kics,
            'test_kics': test_kics
        }
        
        # Log split information
        logger.info(f"Data split summary:")
        logger.info(f"  Train: {len(splits['X_train'])} sequences from {len(train_kics)} KICs")
        logger.info(f"  Val:   {len(splits['X_val'])} sequences from {len(val_kics)} KICs") 
        logger.info(f"  Test:  {len(splits['X_test'])} sequences from {len(test_kics)} KICs")
        
        for split_name in ['train', 'val', 'test']:
            y_split = splits[f'y_{split_name}']
            pos_count = np.sum(y_split)
            total_count = len(y_split)
            logger.info(f"  {split_name.capitalize()} class ratio: {pos_count}/{total_count} = {pos_count/total_count:.3f}")
        
        return splits
    
    def get_preprocessing_stats(self, data_list: List[Tuple[pd.DataFrame, int, str]]) -> Dict:
        """Get statistics about the preprocessing results."""
        stats = {
            'original_light_curves': len(data_list),
            'processed_light_curves': 0,
            'total_sequences': 0,
            'avg_sequences_per_light_curve': 0,
            'rejected_too_short': 0
        }
        
        total_sequences = 0
        processed_count = 0
        
        for df, label, kic_id in data_list:
            sequences = self.process_single_light_curve(df, label, kic_id)
            if sequences:
                processed_count += 1
                total_sequences += len(sequences)
            else:
                stats['rejected_too_short'] += 1
        
        stats['processed_light_curves'] = processed_count
        stats['total_sequences'] = total_sequences
        stats['avg_sequences_per_light_curve'] = total_sequences / processed_count if processed_count > 0 else 0
        
        return stats


if __name__ == "__main__":
    # Example usage and testing
    from data_loader import KeplerDataLoader
    
    # Load data
    loader = KeplerDataLoader()
    data = loader.load_all_data()
    
    if data:
        # Initialize preprocessor
        preprocessor = LightCurvePreprocessor(sequence_length=1000)
        
        # Get preprocessing statistics
        stats = preprocessor.get_preprocessing_stats(data)
        print("Preprocessing Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Process first few light curves for testing
        test_data = data[:5]  # Test with first 5 light curves
        X, y, kic_ids = preprocessor.process_all_data(test_data)
        
        print(f"\nProcessed data shapes:")
        print(f"  X: {X.shape}")
        print(f"  y: {y.shape}")
        print(f"  Unique KICs: {len(set(kic_ids))}")
        
        # Test train/val/test split
        if len(X) > 0:
            splits = preprocessor.create_train_val_test_split(X, y, kic_ids)
            print(f"\nSplit completed successfully!")
    else:
        print("No data found to process.")