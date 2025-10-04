import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import warnings
import argparse
import json
import os
import sys
from pathlib import Path

from cfg.LGBMClassifier_Config import LGBMC_param

# Add ann directory to path for LSTM imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'ann'))

# Import LSTM components (with fallback if TensorFlow not available)
try:
    from ann.data_loader import KeplerDataLoader
    from ann.preprocessing import LightCurvePreprocessor
    from ann.lstm_model import create_simple_model, create_advanced_model
    from ann.utils import ModelEvaluator, PlottingUtils
    from ann.exoplanet_classifier import ExoplanetClassificationPipeline
    LSTM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LSTM components not available: {e}")
    print("Run 'pip install tensorflow astropy' to enable LSTM functionality")
    LSTM_AVAILABLE = False


# Function for cleaning feature names
def clean_feature_names(df):
    """
    Cleans column names of special characters for compatibility with LightGBM and ensures name uniqueness.
    """
    new_columns = []
    name_counts = {}
    
    for col in df.columns:
        # Replace special characters with underscores.
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
        # Remove multiple underscores.
        clean_name = re.sub(r'_+', '_', clean_name)
        # Remove underscores from the start and end." (Or "Strip leading and trailing underscores.
        clean_name = clean_name.strip('_')
        
        # Ensure name uniqueness.
        if clean_name in name_counts:
            name_counts[clean_name] += 1
            unique_name = f"{clean_name}_{name_counts[clean_name]}"
        else:
            name_counts[clean_name] = 0
            unique_name = clean_name
            
        new_columns.append(unique_name)
    
    df.columns = new_columns
    return df

def save_results(results, filename="results.json"):
    """Save results to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {filename}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='NASA Exoplanet Detection Challenge 2025')
    parser.add_argument('--method', type=str, choices=['tsfresh', 'lstm', 'both'], default='tsfresh',
                        help='ML method to use: tsfresh+LightGBM, LSTM, or both')
    parser.add_argument('--num_stars', type=int, default=1000, help='Number of stars to simulate')
    parser.add_argument('--time_points', type=int, default=500, help='Number of time points per star')
    parser.add_argument('--output', type=str, default='results.json', help='Output file for results')
    parser.add_argument('--save_model', action='store_true', help='Save trained model')
    parser.add_argument('--use_real_data', action='store_true', help='Use real Kepler FITS data for LSTM')
    parser.add_argument('--lstm_epochs', type=int, default=50, help='Number of LSTM training epochs')
    parser.add_argument('--lstm_model_type', type=str, choices=['simple', 'advanced'], default='simple',
                        help='LSTM model architecture type')
    return parser.parse_args()

def run_tsfresh_pipeline(args):
    """Run the traditional tsfresh + LightGBM pipeline"""
    print("\n🔬 Running tsfresh + LightGBM Pipeline")
    print("=" * 50)
    
    # Suppress tsfresh warnings about features with non-finite values
    warnings.filterwarnings('ignore', message='.*did not have any finite values.*')
    
    print("📊 Generating synthetic exoplanet data...")
    # Create synthetic data
    num_stars = args.num_stars
    time_points = args.time_points
    data = []
    target_labels = {}

    for star_id in range(num_stars):
        flux = np.random.rand(time_points) * 0.1 + 1.0  # Базовий потік
        
        # Імітація транзиту для половини зірок
        is_exoplanet = np.random.choice([0, 1], p=[0.5, 0.5])
        target_labels[star_id] = is_exoplanet
        
        if is_exoplanet:
            # Імітуємо падіння яскравості (транзит)
            start_time = np.random.randint(50, 400)
            end_time = start_time + 10
            flux[start_time:end_time] -= 0.05
        
        star_df = pd.DataFrame({'id': star_id, 'time': range(time_points), 'flux': flux})
        data.append(star_df)

    full_df = pd.concat(data, ignore_index=True)
    y = pd.Series(target_labels)

    print(f"✅ Generated data for {num_stars} stars")
    print(f"   - Exoplanets: {y.sum()}")
    print(f"   - No exoplanets: {len(y) - y.sum()}")
    
    print("🔬 Extracting features with tsfresh...")
    settings = EfficientFCParameters()
    X = extract_features(full_df, 
                         column_id='id', 
                         column_sort='time', 
                         default_fc_parameters=settings,
                         disable_progressbar=True)

    # Заповнення пропущених значень (якщо такі є)
    impute(X)

    # Clean feature names for LightGBM compatibility
    X = clean_feature_names(X)
    print(f"✅ Extracted {X.shape[1]} features")
    
    print("🤖 Training LightGBM model...")
    
    # Prepare data for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train LightGBM model
    lgb_model = lgb.LGBMClassifier(**LGBMC_param)
    lgb_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lgb_model.predict(X_test)
    y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ tsfresh + LightGBM training completed!")
    print(f"   - Accuracy: {accuracy:.4f}")
    print(f"   - Features used: {X.shape[1]}")
    print(f"   - Test samples: {len(y_test)}")
    
    # Detailed classification report
    print("\n📊 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Exoplanet', 'Exoplanet']))
    
    # Prepare results
    tsfresh_results = {
        'method': 'tsfresh_lightgbm',
        'configuration': {
            'num_stars': num_stars,
            'time_points': time_points,
            'features_extracted': int(X.shape[1]),
            'test_size': len(y_test)
        },
        'performance': {
            'accuracy': float(accuracy),
            'exoplanets_in_test': int(y_test.sum()),
            'no_exoplanets_in_test': int(len(y_test) - y_test.sum()),
            'true_positives': int(((y_pred == 1) & (y_test == 1)).sum()),
            'false_positives': int(((y_pred == 1) & (y_test == 0)).sum()),
            'true_negatives': int(((y_pred == 0) & (y_test == 0)).sum()),
            'false_negatives': int(((y_pred == 0) & (y_test == 1)).sum())
        },
        'feature_importance': {
            'top_10_features': [
                {
                    'feature': feature,
                    'importance': float(importance)
                }
                for feature, importance in zip(
                    X.columns[np.argsort(lgb_model.feature_importances_)[::-1][:10]],
                    np.sort(lgb_model.feature_importances_)[::-1][:10]
                )
            ]
        }
    }
    
    # Save model if requested
    if args.save_model:
        import joblib
        model_filename = 'tsfresh_exoplanet_model.pkl'
        joblib.dump(lgb_model, model_filename)
        print(f"tsfresh model saved to {model_filename}")
    
    return tsfresh_results


def run_lstm_pipeline(args):
    """Run the LSTM neural network pipeline"""
    if not LSTM_AVAILABLE:
        print("❌ LSTM components not available. Install requirements: pip install tensorflow astropy")
        return {'error': 'LSTM components not available'}
    
    print("\n🧠 Running LSTM Neural Network Pipeline")
    print("=" * 50)
    
    # Configure LSTM pipeline
    lstm_config = {
        'epochs': args.lstm_epochs,
        'batch_size': 32,
        'model_type': args.lstm_model_type,
        'sequence_length': min(1000, args.time_points),  # Adapt to available time points
        'save_model': args.save_model,
        'save_plots': True,
        'save_results': True,
        'output_dir': 'lstm_results'
    }
    
    try:
        if args.use_real_data:
            print("📡 Using real Kepler FITS data...")
            lstm_config['data_root'] = 'data'
        else:
            print("📊 Note: Using synthetic data (real FITS data not requested)")
            print("   Use --use_real_data flag to use actual Kepler FITS files")
            # For now, use synthetic approach - could be enhanced to generate FITS-like data
            return {'error': 'Synthetic LSTM data generation not yet implemented. Use --use_real_data flag.'}
        
        # Initialize and run LSTM pipeline
        pipeline = ExoplanetClassificationPipeline(lstm_config)
        lstm_results = pipeline.run_complete_pipeline()
        
        print(f"✅ LSTM training completed!")
        
        return {
            'method': 'lstm',
            'configuration': lstm_config,
            'results': lstm_results
        }
        
    except Exception as e:
        print(f"❌ LSTM pipeline failed: {e}")
        return {'error': str(e)}


def main():
    args = parse_arguments()
    
    print("🌌 NASA Exoplanet Detection Challenge 2025")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Method: {args.method}")
    print(f"  - Number of stars: {args.num_stars}")
    print(f"  - Time points per star: {args.time_points}")
    print(f"  - Output file: {args.output}")
    if args.method in ['lstm', 'both']:
        print(f"  - LSTM epochs: {args.lstm_epochs}")
        print(f"  - LSTM model type: {args.lstm_model_type}")
        print(f"  - Use real FITS data: {args.use_real_data}")
    print()
    
    all_results = {}
    
    # Run selected method(s)
    if args.method in ['tsfresh', 'both']:
        tsfresh_results = run_tsfresh_pipeline(args)
        all_results['tsfresh'] = tsfresh_results
    
    if args.method in ['lstm', 'both']:
        lstm_results = run_lstm_pipeline(args)
        all_results['lstm'] = lstm_results
    
    # Comparison results if both methods were run
    if args.method == 'both':
        print("\n🔍 Method Comparison")
        print("=" * 50)
        
        tsfresh_acc = all_results['tsfresh']['performance']['accuracy'] if 'performance' in all_results['tsfresh'] else 0
        lstm_acc = all_results['lstm']['results']['evaluation']['test_default_threshold']['accuracy'] if 'results' in all_results['lstm'] and 'evaluation' in all_results['lstm']['results'] else 0
        
        print(f"tsfresh + LightGBM Accuracy: {tsfresh_acc:.4f}")
        print(f"LSTM Accuracy: {lstm_acc:.4f}")
        
        if lstm_acc > tsfresh_acc:
            print("🏆 LSTM performed better!")
        elif tsfresh_acc > lstm_acc:
            print("🏆 tsfresh + LightGBM performed better!")
        else:
            print("🤝 Both methods performed equally!")
        
        all_results['comparison'] = {
            'tsfresh_accuracy': tsfresh_acc,
            'lstm_accuracy': lstm_acc,
            'winner': 'lstm' if lstm_acc > tsfresh_acc else 'tsfresh' if tsfresh_acc > lstm_acc else 'tie'
        }
    
    # Save combined results
    save_results(all_results, args.output)
    
    print(f"\n✅ All results saved to {args.output}")
    print("\n🚀 Pipeline execution completed!")


if __name__ == '__main__':
    # Follow Windows multiprocessing best practices
    main()