"""
Data Loader for Kepler FITS Files
Handles loading and initial processing of Kepler light curve data from FITS files.
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeplerDataLoader:
    """
    Loads and processes Kepler light curve data from FITS files.
    
    Handles the three main directories:
    - Positive Class (Confirmed Planets): data/Kepler_confirmed_wget/
    - Negative Class (KOI Candidates): data/Kepler_KOI_wget/
    - Negative Class (Quarterly): data/Kepler_Quarterly_wget/ (if exists)
    """
    
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.confirmed_dir = self.data_root / "Kepler_confirmed_wget"
        self.koi_dir = self.data_root / "Kepler_KOI_wget"
        self.quarterly_dir = self.data_root / "Kepler_Quarterly_wget"
        
        # Feature columns to extract (with alternative names)
        self.required_columns = {
            'time': ['TIME', 'time'],
            'pdcsap_flux': ['PDCSAP_FLUX', 'pdcsap_flux'], 
            'pdcsap_flux_err': ['PDCSAP_FLUX_ERR', 'pdcsap_flux_err'],
            'quality': ['SAP_QUALITY', 'quality', 'QUALITY']
        }
        
    def extract_kic_id(self, filename: str) -> Optional[str]:
        """
        Extract KIC ID from filename.
        Expected format: kplr{KIC_ID}-{timestamp}_llc.fits
        """
        try:
            # Extract KIC ID from filename
            if filename.startswith('kplr') and filename.endswith('.fits'):
                kic_id = filename.split('-')[0].replace('kplr', '')
                return kic_id
            return None
        except Exception as e:
            logger.warning(f"Could not extract KIC ID from {filename}: {e}")
            return None
    
    def read_fits_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Read a single FITS file and extract light curve data.
        
        Returns:
            DataFrame with columns: time, pdcsap_flux, pdcsap_flux_err, quality
        """
        try:
            with fits.open(file_path) as hdul:
                # Light curve data is typically in the first extension
                if len(hdul) > 1:
                    data = hdul[1].data
                else:
                    data = hdul[0].data
                
                # Check if required columns exist (with alternative names)
                if data is None:
                    logger.warning(f"No data found in {file_path}")
                    return None
                
                # Extract columns with flexible naming
                df_dict = {}
                available_columns = [col.upper() for col in data.names]
                
                for target_col, possible_names in self.required_columns.items():
                    found = False
                    for possible_name in possible_names:
                        if possible_name.upper() in available_columns:
                            # Find the actual column name (preserve case)
                            actual_col = next(col for col in data.names if col.upper() == possible_name.upper())
                            df_dict[target_col] = data[actual_col]
                            found = True
                            break
                    
                    if not found:
                        logger.warning(f"Column {target_col} (tried: {possible_names}) not found in {file_path}")
                        return None
                
                df = pd.DataFrame(df_dict)
                
                # Convert to native byte order to avoid big-endian issues
                for col in df.columns:
                    if col != 'kic_id':  # Skip string column
                        df[col] = df[col].astype(np.float64)
                
                # Add metadata
                kic_id = self.extract_kic_id(file_path.name)
                if kic_id:
                    df['kic_id'] = kic_id
                
                return df
                
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None
    
    def get_fits_files(self, directory: Path) -> List[Path]:
        """Get all FITS files from a directory recursively."""
        if not directory.exists():
            logger.warning(f"Directory {directory} does not exist")
            return []
        
        fits_files = []
        for file_path in directory.rglob("*.fits"):
            fits_files.append(file_path)
        
        logger.info(f"Found {len(fits_files)} FITS files in {directory}")
        return fits_files
    
    def load_class_data(self, directory: Path, label: int) -> List[Tuple[pd.DataFrame, int, str]]:
        """
        Load all FITS files from a directory and assign labels.
        
        Returns:
            List of tuples: (dataframe, label, kic_id)
        """
        fits_files = self.get_fits_files(directory)
        loaded_data = []
        
        for file_path in fits_files:
            df = self.read_fits_file(file_path)
            if df is not None and not df.empty:
                kic_id = df['kic_id'].iloc[0] if 'kic_id' in df.columns else "unknown"
                loaded_data.append((df, label, kic_id))
            
        logger.info(f"Successfully loaded {len(loaded_data)} files from {directory}")
        return loaded_data
    
    def load_all_data(self) -> List[Tuple[pd.DataFrame, int, str]]:
        """
        Load all data from all directories.
        
        Returns:
            List of tuples: (dataframe, label, kic_id)
            Label: 1 for confirmed exoplanets, 0 for others
        """
        all_data = []
        
        # Load confirmed exoplanets (label = 1)
        if self.confirmed_dir.exists():
            confirmed_data = self.load_class_data(self.confirmed_dir, label=1)
            all_data.extend(confirmed_data)
            logger.info(f"Loaded {len(confirmed_data)} confirmed exoplanet light curves")
        
        # Load KOI candidates (label = 0)
        if self.koi_dir.exists():
            koi_data = self.load_class_data(self.koi_dir, label=0)
            all_data.extend(koi_data)
            logger.info(f"Loaded {len(koi_data)} KOI candidate light curves")
        
        # Load quarterly data if available (label = 0)
        if self.quarterly_dir.exists():
            quarterly_data = self.load_class_data(self.quarterly_dir, label=0)
            all_data.extend(quarterly_data)
            logger.info(f"Loaded {len(quarterly_data)} quarterly light curves")
        
        logger.info(f"Total loaded: {len(all_data)} light curves")
        return all_data
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of the loaded data."""
        data = self.load_all_data()
        
        positive_count = sum(1 for _, label, _ in data if label == 1)
        negative_count = sum(1 for _, label, _ in data if label == 0)
        
        # Get length statistics
        lengths = [len(df) for df, _, _ in data]
        
        summary = {
            'total_light_curves': len(data),
            'positive_class_count': positive_count,
            'negative_class_count': negative_count,
            'class_ratio': positive_count / len(data) if data else 0,
            'avg_length': np.mean(lengths) if lengths else 0,
            'min_length': np.min(lengths) if lengths else 0,
            'max_length': np.max(lengths) if lengths else 0,
            'median_length': np.median(lengths) if lengths else 0
        }
        
        return summary


if __name__ == "__main__":
    # Example usage
    loader = KeplerDataLoader()
    
    # Get data summary
    summary = loader.get_data_summary()
    print("Data Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Load a small sample for testing
    data = loader.load_all_data()
    if data:
        print(f"\nFirst light curve info:")
        df, label, kic_id = data[0]
        print(f"  KIC ID: {kic_id}")
        print(f"  Label: {label}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Sample data:\n{df.head()}")