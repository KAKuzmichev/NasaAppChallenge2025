"""
Test script for the exoplanet classification pipeline.
Tests data loading and preprocessing without requiring TensorFlow.
"""

import os
import sys

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))

def test_data_loading():
    """Test the data loading functionality."""
    print("=" * 60)
    print("TESTING DATA LOADING")
    print("=" * 60)
    
    try:
        from models.data_loader import KeplerDataLoader
        
        # Initialize loader
        loader = KeplerDataLoader(data_root="../data")
        
        # Get data summary
        summary = loader.get_data_summary()
        print("\nData Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Load actual data
        data = loader.load_all_data()
        print(f"\nLoaded {len(data)} light curves")
        
        if data:
            # Show first light curve info
            df, label, kic_id = data[0]
            print(f"\nFirst light curve sample:")
            print(f"  KIC ID: {kic_id}")
            print(f"  Label: {label}")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Data sample:")
            print(df.head())
            
            return True, data
        else:
            print("No data loaded!")
            return False, []
            
    except Exception as e:
        print(f"Data loading test failed: {e}")
        return False, []

def test_preprocessing(data):
    """Test the preprocessing functionality."""
    print("\n" + "=" * 60)
    print("TESTING DATA PREPROCESSING")
    print("=" * 60)
    
    try:
        from models.preprocessing import LightCurvePreprocessor
        
        # Initialize preprocessor
        preprocessor = LightCurvePreprocessor(
            sequence_length=100,  # Smaller for testing
            stride_ratio=0.5,
            min_length_ratio=0.8
        )
        
        # Test with first few light curves
        test_data = data[:3] if len(data) >= 3 else data
        print(f"\nTesting preprocessing with {len(test_data)} light curves...")
        
        # Get preprocessing stats
        stats = preprocessor.get_preprocessing_stats(test_data)
        print("\nPreprocessing Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Process the data
        X, y, kic_ids = preprocessor.process_all_data(test_data)
        
        print(f"\nProcessed data shapes:")
        print(f"  X: {X.shape}")
        print(f"  y: {y.shape}")
        print(f"  Unique KICs: {len(set(kic_ids))}")
        print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Test train/val/test split
        if len(X) > 0:
            splits = preprocessor.create_train_val_test_split(X, y, kic_ids)
            print(f"\nData split shapes:")
            print(f"  Train: {splits['X_train'].shape}")
            print(f"  Val: {splits['X_val'].shape}")
            print(f"  Test: {splits['X_test'].shape}")
            
            return True, (X, y, kic_ids, splits)
        else:
            print("No sequences generated!")
            return False, None
            
    except Exception as e:
        print(f"Preprocessing test failed: {e}")
        return False, None

def test_utils():
    """Test utility functions."""
    print("\n" + "=" * 60)
    print("TESTING UTILITY FUNCTIONS")
    print("=" * 60)
    
    try:
        from models.utils import ModelEvaluator, DataUtils
        import numpy as np
        
        # Create dummy data for testing
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.binomial(1, 0.3, n_samples)
        y_pred_proba = np.random.beta(2, 5, n_samples)
        y_pred_proba[y_true == 1] += 0.3
        y_pred_proba = np.clip(y_pred_proba, 0, 1)
        
        # Test evaluator
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(y_true, y_pred_proba, "test")
        
        print("Evaluation test completed successfully!")
        
        # Test data utils
        kic_ids = [f"kic_{i//10}" for i in range(n_samples)]  # Multiple sequences per KIC
        analysis = DataUtils.analyze_sequence_distribution(kic_ids)
        
        print("\nSequence distribution analysis:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"Utils test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("EXOPLANET CLASSIFICATION PIPELINE TESTS")
    print("=" * 60)
    
    # Import numpy here since it's needed for testing
    global np
    import numpy as np
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Data Loading
    data_success, data = test_data_loading()
    if data_success:
        success_count += 1
    
    # Test 2: Preprocessing (only if data loading succeeded)
    if data_success and data:
        preprocessing_success, processed = test_preprocessing(data)
        if preprocessing_success:
            success_count += 1
    else:
        print("\nSkipping preprocessing test due to data loading failure")
    
    # Test 3: Utils
    utils_success = test_utils()
    if utils_success:
        success_count += 1
    
    # Final report
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Passed: {success_count}/{total_tests} tests")
    
    if success_count == total_tests:
        print("✅ All tests passed! Pipeline components are working correctly.")
        print("\nNext steps:")
        print("1. Install TensorFlow dependencies: pip install -r requirements.txt")
        print("2. Run the full pipeline: python exoplanet_classifier.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure FITS files are in the correct directories:")
        print("   - data/Kepler_confirmed_wget/")
        print("   - data/Kepler_KOI_wget/")
        print("2. Check that astropy is installed: pip install astropy")
    
    return success_count == total_tests

if __name__ == "__main__":
    main()