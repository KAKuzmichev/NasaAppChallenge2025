# NASA Exoplanet Detection Challenge 2025 - Integrated Pipeline 🌌

This project now combines **two powerful approaches** for exoplanet detection:

1. **Traditional ML**: tsfresh feature extraction + LightGBM classifier
2. **Deep Learning**: LSTM neural networks on real Kepler light curves

## 🚀 Quick Start

### Method 1: Command Line Interface

**Run traditional ML pipeline:**
```bash
python main.py --method tsfresh --num_stars 1000
```

**Run LSTM deep learning pipeline:**
```bash
python main.py --method lstm --use_real_data --lstm_epochs 20
```

**Compare both methods:**
```bash
python main.py --method both --use_real_data --lstm_epochs 20
```

### Method 2: Web Interface

Launch the integrated Streamlit app with both methods:
```bash
streamlit run app.py
```

Navigate to:
- **Data Input**: Configure synthetic or real data
- **Model Training**: Train tsfresh + LightGBM models  
- **LSTM Deep Learning**: Train neural networks on real Kepler data
- **Results Analysis**: Compare both approaches
- **Real Data Explorer**: Explore actual exoplanet catalogs

## 🔧 New Command Line Arguments

- `--method`: Choose 'tsfresh', 'lstm', or 'both'
- `--use_real_data`: Use real Kepler FITS files for LSTM
- `--lstm_epochs`: Number of LSTM training epochs (default: 50)
- `--lstm_model_type`: 'simple' or 'advanced' LSTM architecture

## 📊 Integration Features

### Multi-Method Training
- Train both traditional ML and deep learning models
- Automatic performance comparison
- Unified results output with method comparison

### Real Data Support
- **FITS Files**: Real Kepler telescope light curves
- **CSV Catalogs**: K2, TESS, and Kepler exoplanet databases
- **Preprocessing**: Automatic data quality filtering and normalization

### Advanced Evaluation
- **Cross-Method Comparison**: See which approach performs better
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Visualization**: Training curves, confusion matrices, ROC curves

## 🧠 LSTM Deep Learning Features

### Real Kepler Data Processing
- Loads FITS files from `data/Kepler_confirmed_wget/` and `data/Kepler_KOI_wget/`
- Extracts time, flux, flux_error, and quality columns
- Handles big-endian data format issues automatically

### Advanced Preprocessing
- Quality filtering (removes poor-quality observations)
- Z-score flux normalization per light curve
- Fixed-length sequence generation for LSTM input
- Proper train/validation/test splitting by star ID (prevents data leakage)

### Neural Network Architecture
- **Simple Model**: Single LSTM layer (64 units) + Dense layer
- **Advanced Model**: Bidirectional LSTM layers with higher capacity
- **Regularization**: Dropout, early stopping, learning rate reduction
- **Class Balancing**: Automatic handling of imbalanced datasets

## 🎯 Expected Performance

### Traditional ML (tsfresh + LightGBM)
- **Speed**: Fast training (~1-2 minutes for 1000 stars)
- **Features**: 777+ automatically extracted time-series features
- **Accuracy**: ~85-95% on synthetic data

### Deep Learning (LSTM)
- **Data**: Real Kepler telescope observations
- **Architecture**: End-to-end learning from raw light curves
- **Accuracy**: Comparable or better performance on real data
- **Training Time**: Longer (~5-30 minutes depending on epochs)

## 📈 Results Comparison

When using `--method both`, the system will:

1. Run tsfresh + LightGBM on synthetic data
2. Run LSTM on real Kepler FITS data  
3. Compare accuracies and declare a winner
4. Save comprehensive results to JSON

Example output:
```
🔍 Method Comparison
==========================================
tsfresh + LightGBM Accuracy: 0.8945
LSTM Accuracy: 0.9234
🏆 LSTM performed better!
```

## 🛠️ Installation Requirements

The integrated system requires additional packages:

```bash
# Install all dependencies
pip install -r requirements.txt

# Key additions for LSTM:
pip install tensorflow astropy
```

## 🗂️ Data Directory Structure

```
data/
├── Kepler_confirmed_wget/        # Confirmed exoplanet FITS files
│   └── *.fits
├── Kepler_KOI_wget/             # KOI candidate FITS files  
│   └── *.fits
├── k2pandc_*.csv                # K2 catalog data
├── TOI_*.csv                    # TESS catalog data
└── cumulative_*.csv             # Kepler catalog data
```

## 🌟 Key Advantages of Integration

### 1. **Method Comparison**
- Direct performance comparison between traditional ML and deep learning
- Helps determine the best approach for your specific dataset

### 2. **Real vs Synthetic Data**
- Traditional ML uses synthetic transit simulations
- LSTM uses actual telescope observations
- Compare model performance on different data types

### 3. **Complementary Strengths**
- **tsfresh**: Fast, interpretable, works with limited data
- **LSTM**: Powerful pattern recognition, handles complex signals

### 4. **Production Flexibility**
- Choose the best-performing method for deployment
- Ensemble both methods for improved accuracy

## 🔬 Technical Innovation

### Windows Multiprocessing Support
- Proper `if __name__ == '__main__':` protection for Windows systems
- Follows best practices for cross-platform compatibility

### Robust Error Handling
- Graceful fallback when TensorFlow is not installed
- Clear error messages and installation instructions

### Memory Management
- Efficient data loading for large FITS files
- Configurable batch sizes for different hardware setups

---

**Ready to discover exoplanets with both traditional ML and cutting-edge deep learning!** 🚀🪐