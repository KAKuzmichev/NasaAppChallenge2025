"""
Exoplanet Classifier - Main Training and Evaluation Script
Complete pipeline for training LSTM models on Kepler light curve data.

This script handles the complete workflow:
1. Data loading from FITS files
2. Data preprocessing and sequence generation  
3. Model training with LSTM architecture
4. Model evaluation and results reporting
5. Visualization and saving results

Usage:
    python exoplanet_classifier.py [--config config.json]
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_loader import KeplerDataLoader
    from preprocessing import LightCurvePreprocessor
    from lstm_model import ExoplanetLSTM, create_simple_model, create_advanced_model
    from utils import ModelEvaluator, PlottingUtils, DataUtils
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are available.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('exoplanet_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ExoplanetClassificationPipeline:
    """
    Complete pipeline for exoplanet classification using LSTM on Kepler light curves.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the classification pipeline.
        
        Args:
            config: Configuration dictionary with hyperparameters
        """
        self.config = self._load_default_config()
        if config:
            self.config.update(config)
        
        # Initialize components
        self.data_loader = None
        self.preprocessor = None
        self.model = None
        self.evaluator = ModelEvaluator()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.splits = None
        
        # Results storage
        self.results = {}
        self.training_history = None
        
        logger.info("ExoplanetClassificationPipeline initialized")
        
    def _load_default_config(self) -> dict:
        """Load default configuration parameters."""
        return {
            # Data parameters
            'data_root': '../data',
            'sequence_length': 1000,
            'stride_ratio': 0.5,
            'min_length_ratio': 0.8,
            
            # Model parameters
            'model_type': 'simple',  # 'simple' or 'advanced'
            'lstm_units': 64,
            'dropout_rate': 0.3,
            'dense_units': 32,
            'learning_rate': 0.001,
            'use_bidirectional': False,
            'use_multiple_lstm': False,
            
            # Training parameters
            'epochs': 100,
            'batch_size': 32,
            'validation_split': 0.15,
            'test_split': 0.15,
            'train_split': 0.7,
            'random_state': 42,
            'use_callbacks': True,
            'early_stopping_patience': 10,
            
            # Output parameters
            'save_model': True,
            'save_plots': True,
            'save_results': True,
            'output_dir': 'results',
            'model_filename': 'exoplanet_lstm_model.h5',
            
            # Evaluation parameters
            'classification_threshold': 0.5,
            'find_optimal_threshold': True
        }
    
    def setup_output_directory(self):
        """Create output directory with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config['output_dir']) / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config['output_dir'] = str(output_dir)
        logger.info(f"Output directory created: {output_dir}")
        
        return output_dir
    
    def load_data(self):
        """Load raw data from FITS files."""
        logger.info("Loading data from FITS files...")
        
        self.data_loader = KeplerDataLoader(data_root=self.config['data_root'])
        self.raw_data = self.data_loader.load_all_data()
        
        if not self.raw_data:
            raise ValueError("No data loaded. Check data directory and FITS files.")
        
        # Get data summary
        data_summary = self.data_loader.get_data_summary()
        self.results['data_summary'] = data_summary
        
        logger.info(f"Loaded {len(self.raw_data)} light curves")
        logger.info(f"Positive class: {data_summary['positive_class_count']}")
        logger.info(f"Negative class: {data_summary['negative_class_count']}")
        
        return self.raw_data
    
    def preprocess_data(self):
        """Preprocess data and create train/val/test splits."""
        logger.info("Preprocessing data...")
        
        if self.raw_data is None:
            raise ValueError("No raw data available. Call load_data() first.")
        
        # Initialize preprocessor
        self.preprocessor = LightCurvePreprocessor(
            sequence_length=self.config['sequence_length'],
            stride_ratio=self.config['stride_ratio'],
            min_length_ratio=self.config['min_length_ratio']
        )
        
        # Process all data
        X, y, kic_ids = self.preprocessor.process_all_data(self.raw_data)
        
        if len(X) == 0:
            raise ValueError("No sequences generated after preprocessing.")
        
        # Get preprocessing statistics
        preprocessing_stats = self.preprocessor.get_preprocessing_stats(self.raw_data)
        self.results['preprocessing_stats'] = preprocessing_stats
        
        # Create train/val/test splits
        self.splits = self.preprocessor.create_train_val_test_split(
            X, y, kic_ids,
            train_size=self.config['train_split'],
            val_size=self.config['validation_split'],
            test_size=self.config['test_split'],
            random_state=self.config['random_state']
        )
        
        logger.info("Data preprocessing completed")
        return self.splits
    
    def build_model(self):
        """Build the LSTM model."""
        logger.info("Building LSTM model...")
        
        if self.config['model_type'] == 'advanced':
            self.model = create_advanced_model(
                sequence_length=self.config['sequence_length'],
                n_features=2  # pdcsap_flux_norm and pdcsap_flux_err_norm
            )
        else:
            self.model = ExoplanetLSTM(
                sequence_length=self.config['sequence_length'],
                n_features=2,
                lstm_units=self.config['lstm_units'],
                dropout_rate=self.config['dropout_rate'],
                dense_units=self.config['dense_units'],
                learning_rate=self.config['learning_rate']
            )
            
            self.model.build_model(
                use_bidirectional=self.config['use_bidirectional'],
                use_multiple_lstm=self.config['use_multiple_lstm']
            )
        
        if self.model and self.model.model:
            logger.info(f"Model built with {self.model.model.count_params()} parameters")
        logger.info("Model architecture:")
        logger.info(self.model.get_model_summary())
        
        return self.model
    
    def train_model(self):
        """Train the LSTM model."""
        logger.info("Training model...")
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if self.splits is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        # Train the model
        self.training_history = self.model.train(
            X_train=self.splits['X_train'],
            y_train=self.splits['y_train'],
            X_val=self.splits['X_val'],
            y_val=self.splits['y_val'],
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            use_callbacks=self.config['use_callbacks'],
            verbose=1
        )
        
        logger.info("Model training completed")
        return self.training_history
    
    def evaluate_model(self):
        """Evaluate the trained model on all data splits."""
        logger.info("Evaluating model...")
        
        if self.model is None or self.splits is None:
            raise ValueError("Model not trained or data not available.")
        
        evaluation_results = {}
        
        # Evaluate on all splits
        for split_name in ['train', 'val', 'test']:
            X_split = self.splits[f'X_{split_name}']
            y_split = self.splits[f'y_{split_name}']
            
            # Get predictions
            y_pred_proba = self.model.predict(X_split)
            
            # Evaluate
            split_results = self.evaluator.evaluate_model(
                y_split, y_pred_proba, split_name
            )
            evaluation_results.update(split_results)
        
        self.results['evaluation'] = evaluation_results
        logger.info("Model evaluation completed")
        
        return evaluation_results
    
    def create_visualizations(self):
        """Create and save visualization plots."""
        logger.info("Creating visualizations...")
        
        output_dir = Path(self.config['output_dir'])
        plotter = PlottingUtils()
        
        # Training history plots
        if self.training_history:
            if self.model:
                history_data = self.model.get_training_plots_data()
                plotter.plot_training_history(
                    history_data, 
                    save_path=str(output_dir / "training_history.png")
                )
        
        # Evaluation plots for test set
        if self.splits and self.model:
            X_test = self.splits['X_test']
            y_test = self.splits['y_test']
            y_pred_proba = self.model.predict(X_test)
            y_pred_binary = (y_pred_proba >= self.config['classification_threshold']).astype(int)
            
            # Confusion matrix
            plotter.plot_confusion_matrix(
                y_test, y_pred_binary,
                save_path=str(output_dir / "confusion_matrix.png")
            )
            
            # ROC curve
            plotter.plot_roc_curve(
                y_test, y_pred_proba,
                save_path=str(output_dir / "roc_curve.png")
            )
            
            # Precision-Recall curve
            plotter.plot_precision_recall_curve(
                y_test, y_pred_proba,
                save_path=str(output_dir / "precision_recall_curve.png")
            )
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def save_results(self):
        """Save all results and model."""
        output_dir = Path(self.config['output_dir'])
        
        # Save evaluation results
        if self.config['save_results']:
            results_path = output_dir / "evaluation_results.json"
            self.evaluator.save_results(str(results_path))
        
        # Save model
        if self.config['save_model'] and self.model:
            model_path = output_dir / self.config['model_filename']
            self.model.save_model(str(model_path))
        
        # Save configuration
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Create summary report
        if 'data_summary' in self.results and 'preprocessing_stats' in self.results:
            data_utils = DataUtils()
            summary_report = data_utils.create_summary_report(
                self.results['data_summary'],
                self.results['preprocessing_stats'],
                self.results.get('evaluation', {}),
                save_path=str(output_dir / "summary_report.txt")
            )
            
            print("\n" + summary_report)
        
        logger.info(f"All results saved to {output_dir}")
    
    def run_complete_pipeline(self):
        """Run the complete classification pipeline."""
        logger.info("Starting complete exoplanet classification pipeline...")
        
        try:
            # Setup
            self.setup_output_directory()
            
            # Data pipeline
            self.load_data()
            self.preprocess_data()
            
            # Model pipeline
            self.build_model()
            self.train_model()
            self.evaluate_model()
            
            # Output pipeline
            if self.config['save_plots']:
                self.create_visualizations()
            
            self.save_results()
            
            logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise
        
        return self.results


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Train LSTM model for exoplanet classification"
    )
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='../data',
        help='Root directory containing FITS files'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['simple', 'advanced'],
        default='simple',
        help='Model architecture type'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config_from_file(args.config)
    
    # Override with command line arguments
    if args.data_root:
        config['data_root'] = args.data_root
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.model_type:
        config['model_type'] = args.model_type
    
    # Initialize and run pipeline
    pipeline = ExoplanetClassificationPipeline(config)
    results = pipeline.run_complete_pipeline()
    
    print("\n" + "="*60)
    print("EXOPLANET CLASSIFICATION COMPLETED")
    print("="*60)
    print(f"Results saved to: {pipeline.config['output_dir']}")
    print("Check the summary report for detailed results.")


if __name__ == "__main__":
    main()