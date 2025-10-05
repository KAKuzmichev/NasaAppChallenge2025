"""
Test script to verify the ModelCheckpoint filepath fix is working.
"""

import sys
import os
import numpy as np

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))

try:
    from models.lstm_model import ExoplanetLSTM
    import tempfile
    import shutil
    
    def test_model_checkpoint_fix():
        """Test that ModelCheckpoint uses correct .weights.h5 filepath."""
        print("🧪 Testing ModelCheckpoint filepath fix...")
        
        # Create a simple model
        model = ExoplanetLSTM(
            sequence_length=100,
            n_features=2,
            lstm_units=32,
            dropout_rate=0.3,
            dense_units=16,
            learning_rate=0.001
        )
        
        # Build the model
        model.build_model()
        
        # Create callbacks (this should include the fixed ModelCheckpoint)
        callbacks = model.create_callbacks(save_best_weights=True)
        
        # Check if ModelCheckpoint is in callbacks
        checkpoint_callback = None
        for callback in callbacks:
            if hasattr(callback, 'filepath') and 'ModelCheckpoint' in str(type(callback)):
                checkpoint_callback = callback
                break
        
        if checkpoint_callback:
            print(f"✅ Found ModelCheckpoint callback")
            print(f"📁 Filepath: {checkpoint_callback.filepath}")
            print(f"💾 Save weights only: {checkpoint_callback.save_weights_only}")
            
            # Check if filepath ends with .weights.h5 when save_weights_only=True
            if checkpoint_callback.save_weights_only:
                if checkpoint_callback.filepath.endswith('.weights.h5'):
                    print("✅ Filepath correctly ends with '.weights.h5'")
                    print("🎉 ModelCheckpoint fix is working correctly!")
                    return True
                else:
                    print(f"❌ Filepath should end with '.weights.h5' but ends with '{checkpoint_callback.filepath}'")
                    return False
            else:
                print("ℹ️  save_weights_only is False, .weights.h5 extension not required")
                return True
        else:
            print("❌ ModelCheckpoint callback not found")
            return False
    
    def test_simple_training():
        """Test that the model can train without the filepath error."""
        print("\n🏋️ Testing simple model training...")
        
        try:
            # Create dummy data
            np.random.seed(42)
            batch_size = 8
            sequence_length = 100
            n_features = 2
            
            X_train = np.random.randn(batch_size, sequence_length, n_features)
            y_train = np.random.randint(0, 2, batch_size)
            X_val = np.random.randn(batch_size, sequence_length, n_features)
            y_val = np.random.randint(0, 2, batch_size)
            
            # Create model
            model = ExoplanetLSTM(
                sequence_length=sequence_length,
                n_features=n_features,
                lstm_units=16,
                dropout_rate=0.2,
                dense_units=8,
                learning_rate=0.01
            )
            
            model.build_model()
            
            # Try training for just 1 epoch to verify no errors
            print("🔄 Training for 1 epoch...")
            history = model.train(
                X_train, y_train,
                X_val, y_val,
                epochs=1,
                batch_size=4,
                use_callbacks=True,
                verbose=0
            )
            
            print("✅ Training completed without errors!")
            print("🎉 ModelCheckpoint fix verified through actual training!")
            
            # Clean up any checkpoint files
            import glob
            checkpoint_files = glob.glob("best_exoplanet_model.weights.h5*")
            for file in checkpoint_files:
                try:
                    os.remove(file)
                    print(f"🧹 Cleaned up: {file}")
                except:
                    pass
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed with error: {e}")
            if "save_weights_only" in str(e) and ".weights.h5" in str(e):
                print("❌ This suggests the ModelCheckpoint fix was not applied correctly")
            return False
    
    if __name__ == "__main__":
        print("🔧 Testing ModelCheckpoint Fix")
        print("=" * 50)
        
        # Test 1: Check callback configuration
        test1_passed = test_model_checkpoint_fix()
        
        # Test 2: Try actual training
        test2_passed = test_simple_training()
        
        print("\n📊 Test Results:")
        print("=" * 30)
        print(f"✅ Callback Config Test: {'PASSED' if test1_passed else 'FAILED'}")
        print(f"✅ Training Test: {'PASSED' if test2_passed else 'FAILED'}")
        
        if test1_passed and test2_passed:
            print("\n🎉 All tests passed! The ModelCheckpoint fix is working correctly.")
            print("✅ You can now run the full LSTM pipeline without the filepath error.")
        else:
            print("\n❌ Some tests failed. The ModelCheckpoint issue may not be fully resolved.")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure TensorFlow and other dependencies are installed:")
    print("   pip install tensorflow astropy")
    sys.exit(1)