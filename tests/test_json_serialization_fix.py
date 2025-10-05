"""
Test script to verify JSON serialization fix for NumPy types.
"""

import sys
import os
import numpy as np
import tempfile

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))

try:
    from models.utils import ModelEvaluator, convert_numpy_types
    import json
    
    def test_numpy_conversion():
        """Test the convert_numpy_types function."""
        print("🧪 Testing NumPy Type Conversion")
        print("=" * 40)
        
        # Create test data with various NumPy types
        test_data = {
            'int64_value': np.int64(42),
            'float64_value': np.float64(3.14159),
            'numpy_array': np.array([1, 2, 3, 4, 5]),
            'nested_dict': {
                'inner_int': np.int32(100),
                'inner_float': np.float32(2.718),
                'inner_array': np.array([0.1, 0.2, 0.3])
            },
            'list_with_numpy': [np.int64(1), np.float64(2.5), "string"],
            'regular_values': {
                'normal_int': 42,
                'normal_float': 3.14,
                'normal_string': "test"
            }
        }
        
        print("📊 Original data types:")
        for key, value in test_data.items():
            print(f"  {key}: {type(value)} = {value}")
        
        # Convert using our function
        converted_data = convert_numpy_types(test_data)
        
        print(f"\n✅ Converted data types:")
        for key, value in converted_data.items():
            print(f"  {key}: {type(value)} = {value}")
        
        # Test JSON serialization
        try:
            json_string = json.dumps(converted_data, indent=2)
            print(f"\n✅ JSON serialization successful!")
            print(f"📄 JSON preview (first 200 chars):")
            print(json_string[:200] + "..." if len(json_string) > 200 else json_string)
            return True
        except Exception as e:
            print(f"\n❌ JSON serialization failed: {e}")
            return False
    
    def test_model_evaluator_save():
        """Test ModelEvaluator save_results with NumPy data."""
        print(f"\n🤖 Testing ModelEvaluator JSON Saving")
        print("=" * 45)
        
        # Create evaluator and add some test results with NumPy types
        evaluator = ModelEvaluator()
        
        # Simulate evaluation results with NumPy types
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred_proba = np.random.beta(2, 5, 100)
        
        # Run evaluation (this creates results with NumPy types)
        evaluation_results = evaluator.evaluate_model(y_true, y_pred_proba, "test")
        
        print(f"📊 Generated evaluation results with keys: {list(evaluation_results.keys())}")
        
        # Test saving to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_filepath = temp_file.name
        
        try:
            print(f"💾 Attempting to save results to: {temp_filepath}")
            evaluator.save_results(temp_filepath)
            
            # Verify the file was created and contains valid JSON
            with open(temp_filepath, 'r') as f:
                loaded_data = json.load(f)
            
            print(f"✅ Save successful!")
            print(f"📄 Saved data keys: {list(loaded_data.keys())}")
            print(f"📊 Results keys: {list(loaded_data['results'].keys())}")
            
            # Clean up
            os.unlink(temp_filepath)
            
            return True
            
        except Exception as e:
            print(f"❌ Save failed: {e}")
            # Clean up on failure
            try:
                os.unlink(temp_filepath)
            except:
                pass
            return False
    
    def test_specific_numpy_types():
        """Test conversion of specific NumPy types that commonly cause issues."""
        print(f"\n🔬 Testing Specific NumPy Types")
        print("=" * 35)
        
        problematic_types = {
            'np.int8': np.int8(127),
            'np.int16': np.int16(32767),
            'np.int32': np.int32(2147483647),
            'np.int64': np.int64(9223372036854775807),
            'np.uint8': np.uint8(255),
            'np.uint16': np.uint16(65535),
            'np.uint32': np.uint32(4294967295),
            'np.uint64': np.uint64(18446744073709551615),
            'np.float16': np.float16(3.14),
            'np.float32': np.float32(3.14159),
            'np.float64': np.float64(3.141592653589793),
            'np.bool_': np.bool_(True),
            'np.array_1d': np.array([1, 2, 3]),
            'np.array_2d': np.array([[1, 2], [3, 4]]),
            'np.zeros': np.zeros(5),
            'np.ones': np.ones((2, 3)),
        }
        
        success_count = 0
        total_count = len(problematic_types)
        
        for type_name, value in problematic_types.items():
            try:
                converted = convert_numpy_types(value)
                json.dumps(converted)  # Test serialization
                print(f"  ✅ {type_name}: {type(value)} → {type(converted)}")
                success_count += 1
            except Exception as e:
                print(f"  ❌ {type_name}: Failed - {e}")
        
        print(f"\n📊 Conversion Summary: {success_count}/{total_count} types converted successfully")
        return success_count == total_count
    
    if __name__ == "__main__":
        print("🔧 Testing JSON Serialization Fix for NumPy Types")
        print("=" * 55)
        
        # Run tests
        test1_passed = test_numpy_conversion()
        test2_passed = test_model_evaluator_save()
        test3_passed = test_specific_numpy_types()
        
        print(f"\n📊 Test Results Summary:")
        print("=" * 30)
        print(f"✅ NumPy Conversion Test: {'PASSED' if test1_passed else 'FAILED'}")
        print(f"✅ ModelEvaluator Save Test: {'PASSED' if test2_passed else 'FAILED'}")
        print(f"✅ Specific Types Test: {'PASSED' if test3_passed else 'FAILED'}")
        
        if test1_passed and test2_passed and test3_passed:
            print(f"\n🎉 All tests passed! JSON serialization fix is working correctly.")
            print(f"✅ Your 'Object of type int64 is not JSON serializable' error should be resolved!")
            print(f"\n💡 Key improvements:")
            print(f"   • Added convert_numpy_types() function for type conversion")
            print(f"   • Enhanced save_results() with error handling") 
            print(f"   • Robust fallback mechanism for edge cases")
            print(f"   • Support for nested data structures with NumPy types")
        else:
            print(f"\n❌ Some tests failed. Please check the output above for details.")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure the ann directory and dependencies are available.")
    sys.exit(1)