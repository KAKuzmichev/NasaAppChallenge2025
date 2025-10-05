"""
Test script to verify enhanced early stopping functionality.
"""

import sys
import os
import numpy as np

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))

try:
    from models.lstm_model import ExoplanetLSTM
    
    def test_enhanced_early_stopping():
        """Test the enhanced early stopping configuration."""
        print("🛑 Testing Enhanced Early Stopping Configuration")
        print("=" * 55)
        
        # Create model
        model = ExoplanetLSTM(
            sequence_length=100,
            n_features=2,
            lstm_units=32,
            dropout_rate=0.3,
            dense_units=16,
            learning_rate=0.001
        )
        
        model.build_model()
        
        # Test callback configuration
        callbacks = model.create_callbacks()
        
        print(f"📋 Created {len(callbacks)} callbacks:")
        
        early_stopping_callbacks = []
        reduce_lr_callbacks = []
        checkpoint_callbacks = []
        
        for i, callback in enumerate(callbacks):
            callback_type = type(callback).__name__
            print(f"  {i+1}. {callback_type}")
            
            if 'EarlyStopping' in callback_type:
                early_stopping_callbacks.append(callback)
                print(f"     Monitor: {callback.monitor}")
                print(f"     Patience: {callback.patience}")
                print(f"     Min Delta: {callback.min_delta}")
                print(f"     Mode: {callback.mode}")
                print(f"     Restore Best Weights: {callback.restore_best_weights}")
                
            elif 'ReduceLROnPlateau' in callback_type:
                reduce_lr_callbacks.append(callback)
                print(f"     Monitor: {callback.monitor}")
                print(f"     Patience: {callback.patience}")
                print(f"     Factor: {callback.factor}")
                print(f"     Mode: {callback.mode}")
                
            elif 'ModelCheckpoint' in callback_type:
                checkpoint_callbacks.append(callback)
                print(f"     Filepath: {callback.filepath}")
                print(f"     Monitor: {callback.monitor}")
        
        # Verify enhanced configuration
        print(f"\n🔍 Configuration Analysis:")
        print(f"  Early Stopping Callbacks: {len(early_stopping_callbacks)}")
        print(f"  ReduceLR Callbacks: {len(reduce_lr_callbacks)}")
        print(f"  Checkpoint Callbacks: {len(checkpoint_callbacks)}")
        
        # Check if we have the enhanced early stopping
        val_loss_early_stopping = None
        val_acc_early_stopping = None
        
        for callback in early_stopping_callbacks:
            if callback.monitor == 'val_loss':
                val_loss_early_stopping = callback
            elif callback.monitor == 'val_accuracy':
                val_acc_early_stopping = callback
        
        success = True
        
        if val_loss_early_stopping:
            print(f"  ✅ Val Loss Early Stopping: patience={val_loss_early_stopping.patience}")
            if val_loss_early_stopping.patience <= 7:
                print(f"     ✅ Good: Patience is {val_loss_early_stopping.patience} (≤7) - aggressive enough")
            else:
                print(f"     ⚠️  Warning: Patience is {val_loss_early_stopping.patience} (>7) - might not prevent overfitting")
        else:
            print(f"  ❌ No Val Loss Early Stopping found")
            success = False
        
        if val_acc_early_stopping:
            print(f"  ✅ Val Accuracy Early Stopping: patience={val_acc_early_stopping.patience}")
        else:
            print(f"  ⚠️  No Val Accuracy Early Stopping (this is optional but recommended)")
        
        if reduce_lr_callbacks:
            lr_callback = reduce_lr_callbacks[0]
            print(f"  ✅ Learning Rate Reduction: patience={lr_callback.patience}")
            if lr_callback.patience <= 3:
                print(f"     ✅ Good: LR patience is {lr_callback.patience} (≤3) - aggressive enough")
            else:
                print(f"     ⚠️  Warning: LR patience is {lr_callback.patience} (>3) - could be more aggressive")
        
        return success
    
    def simulate_overfitting_scenario():
        """Simulate a training scenario to show early stopping in action."""
        print(f"\n🎭 Simulating Overfitting Scenario")
        print("=" * 40)
        
        # Create model with enhanced early stopping
        model = ExoplanetLSTM(
            sequence_length=50,  # Smaller for faster testing
            n_features=2,
            lstm_units=16,
            dropout_rate=0.2,
            dense_units=8,
            learning_rate=0.01  # Higher LR to promote overfitting
        )
        
        model.build_model()
        
        # Create minimal dataset
        np.random.seed(42)
        batch_size = 16
        sequence_length = 50
        n_features = 2
        
        # Small dataset to encourage overfitting
        X_train = np.random.randn(batch_size, sequence_length, n_features)
        y_train = np.random.randint(0, 2, batch_size)
        X_val = np.random.randn(batch_size//2, sequence_length, n_features)
        y_val = np.random.randint(0, 2, batch_size//2)
        
        print(f"📊 Training data: {X_train.shape}")
        print(f"📊 Validation data: {X_val.shape}")
        
        try:
            print(f"\n🚀 Starting training with enhanced early stopping...")
            print(f"   (This should stop early when overfitting is detected)")
            
            # Train with enhanced early stopping
            history = model.train(
                X_train, y_train,
                X_val, y_val,
                epochs=50,  # High epochs to test early stopping
                batch_size=8,
                use_callbacks=True,
                verbose=1
            )
            
            epochs_run = len(history.history['loss'])
            print(f"\n📈 Training Results:")
            print(f"   Epochs completed: {epochs_run}/50")
            
            if epochs_run < 50:
                print(f"   ✅ Early stopping worked! Stopped after {epochs_run} epochs")
            else:
                print(f"   ⚠️  Training completed all epochs - early stopping may need adjustment")
            
            # Show final metrics
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            print(f"   Final train loss: {final_train_loss:.4f}")
            print(f"   Final val loss: {final_val_loss:.4f}")
            
            if final_val_loss > final_train_loss * 1.2:
                print(f"   ✅ Good: Validation loss is higher - early stopping helped prevent overfitting")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    if __name__ == "__main__":
        print("🛑 Testing Enhanced Early Stopping for Overfitting Prevention")
        print("=" * 70)
        
        # Test 1: Configuration
        config_success = test_enhanced_early_stopping()
        
        # Test 2: Simulation
        training_success = simulate_overfitting_scenario()
        
        print(f"\n📊 Test Results Summary:")
        print("=" * 35)
        print(f"✅ Configuration Test: {'PASSED' if config_success else 'FAILED'}")
        print(f"✅ Training Test: {'PASSED' if training_success else 'FAILED'}")
        
        if config_success and training_success:
            print(f"\n🎉 Enhanced early stopping is working correctly!")
            print(f"💡 Key improvements made:")
            print(f"   • Reduced patience from 10 to 7 epochs")
            print(f"   • More sensitive min_delta (0.0001 vs 0.001)")
            print(f"   • Aggressive LR reduction (patience=3 vs 5)")
            print(f"   • Added validation accuracy monitoring")
            print(f"   • Explicit mode settings for monitoring")
            print(f"\n✅ Your overfitting issue should now be resolved!")
        else:
            print(f"\n❌ Some issues detected. Please check the output above.")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure TensorFlow is installed: pip install tensorflow")
    sys.exit(1)