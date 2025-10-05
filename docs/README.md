# NASA Exoplanet Detection Challenge 2025 🌌

Advanced machine learning pipeline for detecting exoplanets using time-series analysis with tsfresh and LightGBM.

## Overview

This project provides both **command-line** and **web-based interfaces** for training and evaluating exoplanet detection models using synthetic light curve data.

## Features

- ✨ **Synthetic Data Generation**: Create realistic exoplanet transit light curves
- 🔬 **Feature Extraction**: Use tsfresh to extract time-series features automatically
- 🤖 **Machine Learning**: Train LightGBM models for binary classification
- 📊 **Interactive Web Interface**: Streamlit-based GUI for data exploration and model training
- 💾 **Results Export**: JSON output with detailed performance metrics
- 🎯 **Real Data Support**: Explore actual K2 exoplanet data

## Installation

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 🖥️ Command Line Interface

Run the basic model training:
```bash
python main.py
```

**Advanced options**:
```bash
# Custom configuration
python main.py --num_stars 2000 --time_points 1000 --output my_results.json

# Save the trained model
python main.py --save_model

# Full example
python main.py --num_stars 1500 --time_points 750 --output detailed_results.json --save_model
```

**Parameters:**
- `--num_stars`: Number of stars to simulate (default: 1000)
- `--time_points`: Number of time points per star (default: 500) 
- `--output`: Results output file (default: results.json)
- `--save_model`: Save the trained model as .pkl file

### 🌐 Web Interface

Launch the interactive Streamlit app:
```bash
streamlit run app.py
```

**Web Interface Features:**

#### 📊 Data Input Page
- **Synthetic Data**: Generate customizable exoplanet datasets
- **File Upload**: Upload your own CSV time-series data
- **Real K2 Data**: Explore actual exoplanet observations
- **Interactive Preview**: Visualize sample light curves

#### 🤖 Model Training Page  
- **Real-time Progress**: Watch feature extraction and training progress
- **Quick Metrics**: See accuracy and feature count immediately
- **Automatic Processing**: Handles data preprocessing automatically

#### 📈 Results Analysis Page
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Confusion Matrix**: Interactive confusion matrix visualization
- **Feature Importance**: Top contributing features for predictions
- **Probability Distributions**: Model confidence analysis
- **Model Export**: Save trained models for later use

#### 🔍 Real Data Explorer Page
- **K2 Dataset**: Browse confirmed exoplanets and candidates
- **Interactive Filters**: Filter by status, discovery method, etc.
- **Visualizations**: Distribution charts and data insights

## Output Format

The command-line interface generates structured JSON results:

```json
{
  "configuration": {
    "num_stars": 1000,
    "time_points": 500,
    "features_extracted": 777,
    "test_size": 200
  },
  "performance": {
    "accuracy": 0.8547,
    "exoplanets_in_test": 98,
    "no_exoplanets_in_test": 102,
    "true_positives": 85,
    "false_positives": 12,
    "true_negatives": 90,
    "false_negatives": 13
  },
  "feature_importance": {
    "top_10_features": [
      {
        "feature": "flux_mean",
        "importance": 0.1234
      }
    ]
  }
}
```

## Technical Details

### 🔬 Feature Extraction
- **tsfresh**: Automatic time-series feature extraction
- **777+ features**: Statistical, spectral, and temporal characteristics
- **Robust preprocessing**: Missing value imputation and feature cleaning

### 🤖 Machine Learning
- **LightGBM**: Gradient boosting for fast, accurate classification
- **Cross-validation**: 80/20 train/test split with random state
- **Feature importance**: Identify most predictive characteristics

### 📊 Data Processing
- **Synthetic transits**: Realistic exoplanet signal simulation
- **Noise modeling**: Representative stellar variability
- **Scalable**: Handle hundreds to thousands of stars

## File Structure

```
NASA Challenge 2025/
├── main.py              # Command-line interface
├── app.py               # Streamlit web interface  
├── requirements.txt     # Python dependencies
├── README.md            # This documentation
├── data/                # Data directory
│   └── k2pandc_*.csv    # Real K2 exoplanet data
|   └── TOI_*.csv        # Real TESS exoplanet data
|   └── cumulative_*.csv # Real Kepler exoplanet data
└── results.json         # Output results (generated)
```

## Example Workflows

### Quick Start
```bash
# 1. Basic run with default settings
python main.py

# 2. View results
cat results.json

# 3. Launch web interface for exploration
streamlit run app.py
```

### Research Workflow
```bash
# 1. Generate large dataset
python main.py --num_stars 5000 --time_points 1000 --save_model --output research_results.json

# 2. Analyze results with web interface
streamlit run app.py

# 3. Explore real data comparisons
# (Use Real Data Explorer in web interface)
```

### Custom Data Analysis
```bash
# 1. Prepare your CSV with columns: id, time, flux
# 2. Launch web interface
streamlit run app.py
# 3. Upload data via "Upload CSV File" option
# 4. Train and analyze your model
```

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- tsfresh (feature extraction)
- lightgbm (machine learning)
- streamlit, plotly (web interface)
- matplotlib, seaborn (visualization)

## Performance

**Typical Results:**
- Accuracy: 85-95% on synthetic data
- Feature extraction: 2-5 minutes for 1000 stars
- Training time: < 1 minute
- Real-time web interface updates

## Troubleshooting

### Common Issues

1. **Windows multiprocessing errors**: Fixed with proper `if __name__ == '__main__':` usage
2. **LightGBM feature name errors**: Automatic feature name cleaning implemented
3. **Memory issues**: Reduce `num_stars` or `time_points` for large datasets

### Dependencies
If you encounter import errors, install missing packages:
```bash
pip install package_name
```

## Contributing

This is a NASA Challenge 2025 project focused on exoplanet detection using machine learning techniques.

## License

Educational use for NASA Challenge 2025.