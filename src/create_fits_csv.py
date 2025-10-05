"""
FITS to CSV Converter for Kepler Confirmed Exoplanet Data
Processes all FITS files in Kepler_confirmed_wget directory and creates a unified CSV file.
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path
import logging
from datetime import datetime
import warnings
from tqdm import tqdm

# Suppress astropy warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='astropy')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FitsToCSVConverter:
    """
    Converts Kepler FITS files to a comprehensive CSV dataset.
    """
    
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.confirmed_dir = self.data_root / "Kepler_confirmed_wget"
        
        # Define columns to extract from FITS files
        self.fits_columns = {
            'time': ['TIME', 'time'],
            'pdcsap_flux': ['PDCSAP_FLUX', 'pdcsap_flux'], 
            'pdcsap_flux_err': ['PDCSAP_FLUX_ERR', 'pdcsap_flux_err'],
            'sap_flux': ['SAP_FLUX', 'sap_flux'],
            'sap_flux_err': ['SAP_FLUX_ERR', 'sap_flux_err'],
            'quality': ['SAP_QUALITY', 'quality', 'QUALITY'],
            'pos_corr1': ['POS_CORR1', 'pos_corr1'],
            'pos_corr2': ['POS_CORR2', 'pos_corr2']
        }
        
    def extract_kic_id(self, filename: str) -> str:
        """Extract KIC ID from filename."""
        try:
            if filename.startswith('kplr') and filename.endswith('.fits'):
                return filename.split('-')[0].replace('kplr', '')
            return "unknown"
        except:
            return "unknown"
    
    def extract_quarter_info(self, filename: str) -> str:
        """Extract quarter/timestamp info from filename."""
        try:
            if '-' in filename:
                return filename.split('-')[1].replace('_llc.fits', '')
            return "unknown"
        except:
            return "unknown"
    
    def read_fits_metadata(self, file_path: Path) -> dict:
        """Extract metadata from FITS file header."""
        metadata = {}
        try:
            with fits.open(file_path) as hdul:
                if len(hdul) > 0:
                    header = hdul[0].header
                    
                    # Extract key metadata
                    metadata.update({
                        'object': header.get('OBJECT', ''),
                        'keplerid': header.get('KEPLERID', ''),
                        'ra_obj': header.get('RA_OBJ', np.nan),
                        'dec_obj': header.get('DEC_OBJ', np.nan),
                        'kepmag': header.get('KEPMAG', np.nan),
                        'quarter': header.get('QUARTER', ''),
                        'channel': header.get('CHANNEL', ''),
                        'skygroup': header.get('SKYGROUP', ''),
                        'module': header.get('MODULE', ''),
                        'output': header.get('OUTPUT', ''),
                        'season': header.get('SEASON', ''),
                        'data_rel': header.get('DATA_REL', ''),
                        'obsmode': header.get('OBSMODE', ''),
                        'mission': header.get('MISSION', '')
                    })
        except Exception as e:
            logger.warning(f"Could not extract metadata from {file_path}: {e}")
        
        return metadata
    
    def read_fits_data(self, file_path: Path) -> tuple:
        """
        Read FITS file and extract both data and metadata.
        
        Returns:
            tuple: (DataFrame with light curve data, metadata dict)
        """
        try:
            with fits.open(file_path) as hdul:
                # Get metadata
                metadata = self.read_fits_metadata(file_path)
                
                # Light curve data is typically in the first extension
                if len(hdul) > 1:
                    data = hdul[1].data
                else:
                    data = hdul[0].data
                
                if data is None:
                    return None, metadata
                
                # Extract columns with flexible naming
                df_dict = {}
                available_columns = [col.upper() for col in data.names]
                
                for target_col, possible_names in self.fits_columns.items():
                    found = False
                    for possible_name in possible_names:
                        if possible_name.upper() in available_columns:
                            actual_col = next(col for col in data.names 
                                            if col.upper() == possible_name.upper())
                            df_dict[target_col] = data[actual_col]
                            found = True
                            break
                    
                    # If column not found, create with NaN values
                    if not found:
                        df_dict[target_col] = np.full(len(data), np.nan)
                
                df = pd.DataFrame(df_dict)
                
                # Convert to native byte order
                for col in df.columns:
                    if df[col].dtype.kind in ['f', 'i']:  # float or int
                        df[col] = df[col].astype(np.float64)
                
                return df, metadata
                
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None, {}
    
    def calculate_light_curve_features(self, df: pd.DataFrame) -> dict:
        """Calculate statistical features from light curve data."""
        features = {}
        
        # Focus on PDCSAP flux as it's the cleaned version
        flux_col = 'pdcsap_flux'
        if flux_col in df.columns:
            flux = df[flux_col].dropna()
            
            if len(flux) > 0:
                features.update({
                    'data_points': len(flux),
                    'flux_mean': np.mean(flux),
                    'flux_std': np.std(flux),
                    'flux_median': np.median(flux),
                    'flux_min': np.min(flux),
                    'flux_max': np.max(flux),
                    'flux_range': np.max(flux) - np.min(flux),
                    'flux_var': np.var(flux),
                    'flux_skew': pd.Series(flux).skew(),
                    'flux_kurtosis': pd.Series(flux).kurtosis()
                })
                
                # Time-based features
                if 'time' in df.columns:
                    time_data = df['time'].dropna()
                    if len(time_data) > 1:
                        features.update({
                            'time_span': np.max(time_data) - np.min(time_data),
                            'time_start': np.min(time_data),
                            'time_end': np.max(time_data),
                            'cadence': np.median(np.diff(time_data))
                        })
                
                # Quality flags
                if 'quality' in df.columns:
                    quality = df['quality'].dropna()
                    features.update({
                        'quality_flags_count': np.sum(quality > 0),
                        'quality_good_data_fraction': np.sum(quality == 0) / len(quality) if len(quality) > 0 else 0
                    })
        
        return features
    
    def process_all_fits_files(self, max_files: int = None) -> pd.DataFrame:
        """
        Process all FITS files and create comprehensive dataset.
        
        Args:
            max_files: Limit number of files to process (for testing)
        
        Returns:
            DataFrame with all processed data
        """
        if not self.confirmed_dir.exists():
            raise FileNotFoundError(f"Directory {self.confirmed_dir} does not exist")
        
        # Get all FITS files
        fits_files = list(self.confirmed_dir.glob("*.fits"))
        logger.info(f"Found {len(fits_files)} FITS files")
        
        if max_files:
            fits_files = fits_files[:max_files]
            logger.info(f"Processing first {len(fits_files)} files")
        
        all_records = []
        failed_files = []
        
        # Process files with progress bar
        for file_path in tqdm(fits_files, desc="Processing FITS files"):
            try:
                # Extract basic file info
                kic_id = self.extract_kic_id(file_path.name)
                quarter_info = self.extract_quarter_info(file_path.name)
                
                # Read FITS data and metadata
                df, metadata = self.read_fits_data(file_path)
                
                if df is not None and not df.empty:
                    # Calculate light curve features
                    features = self.calculate_light_curve_features(df)
                    
                    # Create record
                    record = {
                        'kic_id': kic_id,
                        'filename': file_path.name,
                        'quarter_info': quarter_info,
                        'file_size_mb': file_path.stat().st_size / (1024*1024),
                        'confirmed_exoplanet': 1  # All files in confirmed directory
                    }
                    
                    # Add metadata
                    record.update(metadata)
                    
                    # Add calculated features
                    record.update(features)
                    
                    all_records.append(record)
                else:
                    failed_files.append(file_path.name)
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                failed_files.append(file_path.name)
        
        if failed_files:
            logger.warning(f"Failed to process {len(failed_files)} files: {failed_files[:10]}...")
        
        # Create DataFrame
        df_result = pd.DataFrame(all_records)
        logger.info(f"Successfully processed {len(df_result)} files")
        
        return df_result
    
    def save_to_csv(self, df: pd.DataFrame, output_path: str = None) -> str:
        """Save DataFrame to CSV with timestamp."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"kepler_confirmed_exoplanets_{timestamp}.csv"
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Saved dataset to {output_file}")
        logger.info(f"Dataset shape: {df.shape}")
        
        # Print summary
        print(f"\n📊 Dataset Summary:")
        print(f"   Total records: {len(df):,}")
        print(f"   Total columns: {len(df.columns)}")
        print(f"   File size: {output_file.stat().st_size / (1024*1024):.1f} MB")
        print(f"   Unique KIC IDs: {df['kic_id'].nunique():,}")
        
        if 'data_points' in df.columns:
            print(f"   Avg data points per light curve: {df['data_points'].mean():.0f}")
            print(f"   Total data points: {df['data_points'].sum():,}")
        
        return str(output_file)


def main():
    """Main function to process all FITS files."""
    print("🌌 Kepler Confirmed Exoplanet FITS to CSV Converter")
    print("=" * 60)
    
    # Initialize converter
    converter = FitsToCSVConverter()
    
    # Process all files (remove max_files limit for full processing)
    # For testing, you can add max_files=100 parameter
    print("📁 Processing all FITS files in Kepler_confirmed_wget directory...")
    df = converter.process_all_fits_files()
    
    if df.empty:
        print("❌ No data was processed!")
        return
    
    # Save to CSV
    output_file = converter.save_to_csv(df)
    
    print(f"\n✅ Successfully created CSV file: {output_file}")
    
    # Show sample data
    print(f"\n📋 Sample Data (first 5 rows):")
    print(df.head()[['kic_id', 'filename', 'data_points', 'flux_mean', 'flux_std']].to_string(index=False))
    
    print(f"\n📋 All Columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")


if __name__ == "__main__":
    main()