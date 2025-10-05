"""
Test script for FITS to CSV converter - processes just a few files first
"""

from src.create_fits_csv import FitsToCSVConverter
import logging

# Set up logging to see progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_converter():
    """Test the converter with a small sample."""
    print("🧪 Testing FITS to CSV converter with sample files...")
    
    converter = FitsToCSVConverter()
    
    # Process just 5 files for testing
    df = converter.process_all_fits_files(max_files=5)
    
    if not df.empty:
        print(f"\n✅ Test successful! Processed {len(df)} files")
        print(f"Columns: {len(df.columns)}")
        print(f"Sample data:\n{df.head()}")
        
        # Save test file
        test_output = converter.save_to_csv(df, "test_kepler_sample.csv")
        print(f"\n💾 Test file saved as: {test_output}")
        
        return True
    else:
        print("❌ Test failed - no data processed")
        return False

if __name__ == "__main__":
    test_converter()