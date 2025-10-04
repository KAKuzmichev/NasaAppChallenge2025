# Data Files Documentation 📊

## Available Datasets

Your NASA Exoplanet Detection Challenge 2025 project now includes **3 comprehensive exoplanet datasets** totaling **23.0 MB** of real astronomical data:

### 🔹 **K2 Exoplanet Data** 
- **File**: `k2pandc_2025.09.27_13.36.16.csv`
- **Size**: 7.6 MB
- **Columns**: 295 parameters
- **Description**: K2 mission planet and candidate data from NASA Exoplanet Archive
- **Contains**: Confirmed exoplanets and candidates from K2 observations
- **Key Fields**: 
  - Planet names, host stars
  - Orbital parameters (period, radius, mass)
  - Stellar characteristics
  - Discovery information

### 🔹 **TOI (TESS Objects of Interest)**
- **File**: `TOI_2025.09.27_13.34.57.csv` 
- **Size**: 4.2 MB
- **Columns**: 87 parameters
- **Description**: TESS mission objects of interest
- **Contains**: Latest TESS discoveries and candidates
- **Key Fields**:
  - TOI identifiers
  - Transit parameters
  - Stellar properties
  - Disposition status

### 🔹 **Cumulative Exoplanet Data**
- **File**: `cumulative_2025.09.27_13.35.40.csv`
- **Size**: 11.2 MB  
- **Columns**: 141 parameters
- **Description**: Comprehensive cumulative exoplanet archive
- **Contains**: All confirmed exoplanets from various missions
- **Key Fields**:
  - Complete orbital solutions
  - Physical characteristics
  - Discovery methods
  - Publication references

## Data Integration Features

### 🌐 **Web Interface Integration**
The `app.py` Streamlit interface now automatically:
- **Detects available datasets** in the `data/` directory
- **Allows dataset selection** from a dropdown menu
- **Loads the most recent version** of each dataset type
- **Provides data preview** and basic statistics
- **Handles different file formats** and comment lines

### 🖥️ **Command Line Tools**
- **`data_manager.py`**: Comprehensive data management utility
  ```bash
  python data_manager.py --action summary    # Overview of all datasets
  python data_manager.py --action explore    # Interactive data explorer
  python data_manager.py --action check      # Integrity verification
  ```

### 📊 **Automatic Dataset Detection**
The system automatically finds and categorizes data files by pattern:
- `k2pandc_*.csv` → K2 Exoplanet Data
- `TOI_*.csv` → TESS Objects of Interest  
- `cumulative_*.csv` → Cumulative Exoplanet Data

## Usage in Your Project

### **Real Data Explorer Page**
Navigate to the "Real Data Explorer" page in the web interface to:
- Browse confirmed exoplanets vs candidates
- Filter by discovery method, year, etc.
- Visualize planet radius distributions
- Examine stellar host properties
- Export filtered datasets

### **Model Training with Real Data**
While the current ML pipeline uses synthetic light curve data, the real datasets provide:
- **Validation targets** for model predictions
- **Statistical baselines** for comparison
- **Feature engineering insights** from real exoplanet characteristics
- **Publication-quality context** for results

### **Research Applications**
Use these datasets for:
- **Population studies** of exoplanet characteristics
- **Detection bias analysis** across different missions
- **Stellar host correlation** studies
- **Discovery method comparison** research

## File Management

### **Automatic Updates**
The system handles multiple versions by:
- **Timestamp detection** in filenames
- **Automatic selection** of newest files
- **Graceful fallback** if files are missing
- **User notification** of loaded datasets

### **Data Integrity**
Built-in checks verify:
- ✅ File accessibility and format
- ✅ Expected column structure  
- ✅ Reasonable file sizes
- ✅ Data completeness

### **Error Handling**
Robust error handling for:
- Missing data directories
- Corrupted files
- Format inconsistencies
- Memory limitations

## Technical Notes

### **Performance Optimization**
- **Lazy loading**: Only loads data when requested
- **Memory efficient**: Reads headers first for file selection
- **Caching**: Stores loaded datasets in session state
- **Progress tracking**: Shows loading status for large files

### **Compatibility**
- **Cross-platform**: Works on Windows, macOS, Linux
- **Version agnostic**: Handles different CSV formats
- **Encoding robust**: Supports various text encodings
- **Comment handling**: Properly processes NASA archive headers

## Quick Start

1. **Explore your data**:
   ```bash
   python data_manager.py --action summary
   ```

2. **Launch web interface**:
   ```bash
   streamlit run app.py
   ```

3. **Go to "Real Data Explorer"** page and select a dataset

4. **Browse and analyze** real exoplanet discoveries!

Your data files are now fully integrated into the NASA Challenge 2025 interface! 🚀