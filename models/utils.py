"""
Utility functions for exoplanet classification pipeline.
Includes evaluation metrics, plotting functions, and helper utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from typing import Dict, List, Tuple, Optional
import logging
import json
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to native Python types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        Object with NumPy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


class ModelEvaluator:
    """
    Comprehensive evaluation suite for the exoplanet classification model.
    """
    
    def __init__(self):
        self.results = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                         y_pred_binary: Optional[np.ndarray] = None,
                         threshold: float = 0.5) -> Dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            y_pred_binary: Predicted binary labels (optional, will compute from probabilities)
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        if y_pred_binary is None:
            y_pred_binary = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true, y_pred_binary, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0,
            'threshold': threshold,
            'n_samples': len(y_true),
            'n_positive': np.sum(y_true),
            'n_negative': len(y_true) - np.sum(y_true),
            'positive_rate': np.mean(y_true)
        }
        
        # Calculate specificity (true negative rate)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = metrics['recall']  # Same as recall
        
        # Calculate balanced accuracy
        metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
        
        return metrics
    
    def evaluate_model(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      split_name: str = "test") -> Dict:
        """
        Comprehensive model evaluation for a specific data split.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            split_name: Name of the data split (train/val/test)
            
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Evaluating model on {split_name} set...")
        
        # Calculate metrics for default threshold (0.5)
        default_metrics = self.calculate_metrics(y_true, y_pred_proba, threshold=0.5)
        
        # Find optimal threshold based on F1 score
        optimal_threshold = self.find_optimal_threshold(y_true, y_pred_proba)
        optimal_metrics = self.calculate_metrics(y_true, y_pred_proba, threshold=optimal_threshold)
        
        # Store results
        evaluation_results = {
            f'{split_name}_default_threshold': default_metrics,
            f'{split_name}_optimal_threshold': optimal_metrics,
            f'{split_name}_optimal_threshold_value': optimal_threshold
        }
        
        self.results.update(evaluation_results)
        
        # Print summary
        self.print_evaluation_summary(default_metrics, optimal_metrics, split_name)
        
        return evaluation_results
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Find optimal classification threshold based on F1 score.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Optimal threshold value
        """
        thresholds = np.linspace(0.1, 0.9, 81)  # Test thresholds from 0.1 to 0.9
        f1_scores = []
        
        for threshold in thresholds:
            y_pred_binary = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)
            f1_scores.append(f1)
        
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        logger.info(f"Optimal threshold: {optimal_threshold:.3f} (F1: {f1_scores[optimal_idx]:.3f})")
        
        return optimal_threshold
    
    def print_evaluation_summary(self, default_metrics: Dict, optimal_metrics: Dict, 
                                split_name: str):
        """Print a formatted summary of evaluation results."""
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY - {split_name.upper()} SET")
        print(f"{'='*60}")
        
        print(f"Dataset Info:")
        print(f"  Total samples: {default_metrics['n_samples']}")
        print(f"  Positive class: {default_metrics['n_positive']} ({default_metrics['positive_rate']:.1%})")
        print(f"  Negative class: {default_metrics['n_negative']}")
        
        print(f"\nDefault Threshold (0.5):")
        print(f"  Accuracy:    {default_metrics['accuracy']:.4f}")
        print(f"  Precision:   {default_metrics['precision']:.4f}")
        print(f"  Recall:      {default_metrics['recall']:.4f}")
        print(f"  F1 Score:    {default_metrics['f1_score']:.4f}")
        print(f"  ROC AUC:     {default_metrics['roc_auc']:.4f}")
        print(f"  Specificity: {default_metrics['specificity']:.4f}")
        
        print(f"\nOptimal Threshold ({optimal_metrics['threshold']:.3f}):")
        print(f"  Accuracy:    {optimal_metrics['accuracy']:.4f}")
        print(f"  Precision:   {optimal_metrics['precision']:.4f}")
        print(f"  Recall:      {optimal_metrics['recall']:.4f}")
        print(f"  F1 Score:    {optimal_metrics['f1_score']:.4f}")
        print(f"  ROC AUC:     {optimal_metrics['roc_auc']:.4f}")
        print(f"  Specificity: {optimal_metrics['specificity']:.4f}")
    
    def save_results(self, filepath: str):
        """Save evaluation results to JSON file with proper NumPy type conversion."""
        try:
            # Convert NumPy types to JSON-serializable types
            json_safe_results = convert_numpy_types(self.results)
            
            results_with_metadata = {
                'evaluation_timestamp': datetime.now().isoformat(),
                'results': json_safe_results
            }
            
            with open(filepath, 'w') as f:
                json.dump(results_with_metadata, f, indent=2, default=str)
            
            logger.info(f"Evaluation results saved to {filepath}")
            
        except (TypeError, ValueError) as e:
            logger.error(f"Error saving results to JSON: {e}")
            # Fallback: save with string conversion for any remaining issues
            try:
                fallback_results = json.loads(json.dumps(self.results, default=str))
                results_with_metadata = {
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'results': fallback_results
                }
                
                with open(filepath, 'w') as f:
                    json.dump(results_with_metadata, f, indent=2)
                
                logger.info(f"Evaluation results saved to {filepath} (using fallback method)")
                
            except Exception as fallback_error:
                logger.error(f"Failed to save results even with fallback: {fallback_error}")
                raise


class PlottingUtils:
    """
    Utility class for creating evaluation plots and visualizations.
    """
    
    @staticmethod
    def plot_training_history(history_data: Dict, save_path: Optional[str] = None):
        """
        Plot training history (loss, accuracy, etc.).
        
        Args:
            history_data: Dictionary with training metrics
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        epochs = history_data.get('epochs', [])
        
        # Plot loss
        axes[0, 0].plot(epochs, history_data.get('train_loss', []), 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, history_data.get('val_loss', []), 'r-', label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot accuracy
        axes[0, 1].plot(epochs, history_data.get('train_accuracy', []), 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, history_data.get('val_accuracy', []), 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot precision
        if history_data.get('train_precision'):
            axes[1, 0].plot(epochs, history_data.get('train_precision', []), 'b-', label='Training Precision')
            axes[1, 0].plot(epochs, history_data.get('val_precision', []), 'r-', label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot recall
        if history_data.get('train_recall'):
            axes[1, 1].plot(epochs, history_data.get('train_recall', []), 'b-', label='Training Recall')
            axes[1, 1].plot(epochs, history_data.get('val_recall', []), 'r-', label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                             class_names: List[str] = None, 
                             save_path: Optional[str] = None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            save_path: Optional path to save the plot
        """
        if class_names is None:
            class_names = ['Non-Exoplanet', 'Exoplanet']
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                      save_path: Optional[str] = None):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Optional path to save the plot
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   save_path: Optional[str] = None):
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Optional path to save the plot
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        
        # Add baseline
        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='red', linestyle='--', 
                   label=f'Baseline (Random) = {baseline:.3f}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve plot saved to {save_path}")
        
        plt.show()


class DataUtils:
    """
    Utility functions for data handling and analysis.
    """
    
    @staticmethod
    def analyze_sequence_distribution(kic_ids: List[str]) -> Dict:
        """
        Analyze the distribution of sequences per KIC ID.
        
        Args:
            kic_ids: List of KIC IDs for each sequence
            
        Returns:
            Analysis results
        """
        kic_counts = pd.Series(kic_ids).value_counts()
        
        analysis = {
            'total_sequences': len(kic_ids),
            'unique_kics': len(kic_counts),
            'avg_sequences_per_kic': float(kic_counts.mean()),
            'min_sequences_per_kic': int(kic_counts.min()),
            'max_sequences_per_kic': int(kic_counts.max()),
            'median_sequences_per_kic': float(kic_counts.median()),
            'sequences_per_kic_std': float(kic_counts.std())
        }
        
        return analysis
    
    @staticmethod
    def create_summary_report(data_summary: Dict, preprocessing_stats: Dict,
                            evaluation_results: Dict, save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive summary report.
        
        Args:
            data_summary: Data loading summary
            preprocessing_stats: Preprocessing statistics
            evaluation_results: Model evaluation results
            save_path: Optional path to save the report
            
        Returns:
            Report string
        """
        report_lines = []
        report_lines.append("EXOPLANET CLASSIFICATION - SUMMARY REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Data summary
        report_lines.append("DATA SUMMARY:")
        report_lines.append("-" * 20)
        for key, value in data_summary.items():
            report_lines.append(f"  {key}: {value}")
        report_lines.append("")
        
        # Preprocessing summary
        report_lines.append("PREPROCESSING SUMMARY:")
        report_lines.append("-" * 25)
        for key, value in preprocessing_stats.items():
            report_lines.append(f"  {key}: {value}")
        report_lines.append("")
        
        # Model evaluation summary
        report_lines.append("MODEL EVALUATION SUMMARY:")
        report_lines.append("-" * 30)
        for key, value in evaluation_results.items():
            if isinstance(value, dict):
                report_lines.append(f"  {key}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, float):
                        report_lines.append(f"    {subkey}: {subvalue:.4f}")
                    else:
                        report_lines.append(f"    {subkey}: {subvalue}")
            else:
                if isinstance(value, float):
                    report_lines.append(f"  {key}: {value:.4f}")
                else:
                    report_lines.append(f"  {key}: {value}")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Summary report saved to {save_path}")
        
        return report


if __name__ == "__main__":
    # Example usage and testing
    
    # Generate dummy data for testing
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.binomial(1, 0.3, n_samples)  # 30% positive class
    y_pred_proba = np.random.beta(2, 5, n_samples)  # Dummy probabilities
    y_pred_proba[y_true == 1] += 0.3  # Boost probabilities for positive class
    y_pred_proba = np.clip(y_pred_proba, 0, 1)
    
    # Test evaluator
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(y_true, y_pred_proba, "test")
    
    # Test plotting functions
    print("\nTesting plotting functions...")
    
    # Create dummy training history
    epochs = list(range(1, 21))
    history_data = {
        'epochs': epochs,
        'train_loss': [0.7 - 0.02*i + np.random.normal(0, 0.01) for i in epochs],
        'val_loss': [0.7 - 0.015*i + np.random.normal(0, 0.02) for i in epochs],
        'train_accuracy': [0.5 + 0.02*i + np.random.normal(0, 0.01) for i in epochs],
        'val_accuracy': [0.5 + 0.015*i + np.random.normal(0, 0.02) for i in epochs]
    }
    
    plotter = PlottingUtils()
    plotter.plot_training_history(history_data)
    
    # Test other plots
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    plotter.plot_confusion_matrix(y_true, y_pred_binary)
    plotter.plot_roc_curve(y_true, y_pred_proba)
    plotter.plot_precision_recall_curve(y_true, y_pred_proba)
    
    print("All tests completed successfully!")