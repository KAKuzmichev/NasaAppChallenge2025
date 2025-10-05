import streamlit as st
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys

# Add models directory to path for LSTM imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# Import LSTM components (with fallback if TensorFlow not available)
try:
    from models.data_loader import KeplerDataLoader
    from models.preprocessing import LightCurvePreprocessor  
    from models.lstm_model import create_simple_model, create_advanced_model
    from models.utils import ModelEvaluator, PlottingUtils
    from models.exoplanet_classifier import ExoplanetClassificationPipeline
    LSTM_AVAILABLE = True
except ImportError as e:
    st.sidebar.warning(f"LSTM components not available: {e}")
    st.sidebar.info("Run 'pip install tensorflow astropy' to enable LSTM functionality")
    LSTM_AVAILABLE = False


# Suppress warnings
warnings.filterwarnings('ignore', message='.*did not have any finite values.*')

# Function for cleaning feature names
def clean_feature_names(df):
    """
    Очищає назви колонок від спеціальних символів для сумісності з LightGBM
    та забезпечує унікальність назв
    """
    new_columns = []
    name_counts = {}
    
    for col in df.columns:
        # Замінюємо спеціальні символи на підкреслення
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
        # Видаляємо множинні підкреслення
        clean_name = re.sub(r'_+', '_', clean_name)
        # Видаляємо підкреслення з початку та кінця
        clean_name = clean_name.strip('_')
        
        # Забезпечуємо унікальність назв
        if clean_name in name_counts:
            name_counts[clean_name] += 1
            unique_name = f"{clean_name}_{name_counts[clean_name]}"
        else:
            name_counts[clean_name] = 0
            unique_name = clean_name
            
        new_columns.append(unique_name)
    
    df.columns = new_columns
    return df

def generate_synthetic_data(num_stars, time_points, exoplanet_ratio=0.5):
    """Generate synthetic exoplanet detection data"""
    data = []
    target_labels = {}
    
    for star_id in range(num_stars):
        # Base flux with some noise
        flux = np.random.rand(time_points) * 0.1 + 1.0
        
        # Simulate transit for some stars
        is_exoplanet = np.random.choice([0, 1], p=[1-exoplanet_ratio, exoplanet_ratio])
        target_labels[star_id] = is_exoplanet
        
        if is_exoplanet:
            # Simulate brightness dip (transit)
            start_time = np.random.randint(50, time_points-50)
            transit_duration = np.random.randint(5, 15)
            end_time = start_time + transit_duration
            transit_depth = np.random.uniform(0.01, 0.08)
            flux[start_time:end_time] -= transit_depth
        
        star_df = pd.DataFrame({
            'id': star_id, 
            'time': range(time_points), 
            'flux': flux
        })
        data.append(star_df)
    
    return pd.concat(data, ignore_index=True), pd.Series(target_labels)

def train_lstm_model(config=None):
    """Train LSTM model using real Kepler data or configuration"""
    if not LSTM_AVAILABLE:
        st.error("LSTM components not available. Please install TensorFlow and astropy.")
        return None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text('Initializing LSTM pipeline...')
        progress_bar.progress(10)
        
        # Default configuration
        default_config = {
            'data_root': 'data',
            'epochs': 10,  # Reduced for demo
            'batch_size': 16,
            'model_type': 'simple',
            'sequence_length': 500,
            'save_model': True,
            'save_plots': False,  # Disable file saving in Streamlit
            'save_results': False,
            'output_dir': 'temp_results'
        }
        
        if config:
            default_config.update(config)
        
        status_text.text('Loading Kepler data...')
        progress_bar.progress(25)
        
        # Initialize pipeline
        pipeline = ExoplanetClassificationPipeline(default_config)
        
        # Load data
        pipeline.load_data()
        
        status_text.text('Preprocessing light curves...')
        progress_bar.progress(50)
        
        # Preprocess data
        pipeline.preprocess_data()
        
        status_text.text('Building LSTM model...')
        progress_bar.progress(65)
        
        # Build model
        pipeline.build_model()
        
        status_text.text('Training LSTM model...')
        progress_bar.progress(75)
        
        # Train model
        pipeline.train_model()
        
        status_text.text('Evaluating model...')
        progress_bar.progress(90)
        
        # Evaluate
        evaluation_results = pipeline.evaluate_model()
        
        status_text.text('LSTM training complete!')
        progress_bar.progress(100)
        
        return {
            'pipeline': pipeline,
            'results': evaluation_results,
            'model': pipeline.model,
            'splits': pipeline.splits
        }
        
    except Exception as e:
        st.error(f"LSTM training failed: {str(e)}")
        status_text.text(f'Error: {str(e)}')
        return None
    """Train the exoplanet detection model"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
def train_tsfresh_model(df, y):
    """Train the exoplanet detection model using tsfresh + LightGBM"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text('Extracting features with tsfresh...')
    progress_bar.progress(25)
    
    # Feature extraction
    settings = EfficientFCParameters()
    X = extract_features(df, 
                         column_id='id', 
                         column_sort='time', 
                         default_fc_parameters=settings,
                         disable_progressbar=True)
    
    status_text.text('Cleaning feature names...')
    progress_bar.progress(50)
    
    # Fill missing values
    impute(X)
    
    # Clean feature names
    X = clean_feature_names(X)
    
    status_text.text('Training model...')
    progress_bar.progress(75)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = lgb.LGBMClassifier(objective='binary', random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    
    status_text.text('Training complete!')
    progress_bar.progress(100)
    
    return model, X, y, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, accuracy

def load_real_data():
    """Load real exoplanet data from CSV"""
    import glob
    import os
    
    # Try to find available data files
    data_files = {
        'K2 Data': glob.glob('data/k2pandc_*.csv'),
        'TOI Data': glob.glob('data/TOI_*.csv'),
        'Cumulative Data': glob.glob('data/cumulative_*.csv')
    }
    
    # Find the most recent files
    available_files = {}
    for data_type, files in data_files.items():
        if files:
            # Sort by modification time, get the most recent
            latest_file = max(files, key=os.path.getmtime)
            available_files[data_type] = latest_file
    
    if not available_files:
        st.error("No real data files found in the data/ directory.")
        return None, None
    
    # Let user choose which dataset to load
    selected_dataset = st.selectbox(
        "Choose a dataset:",
        list(available_files.keys()),
        help="Select which exoplanet dataset to explore"
    )
    
    try:
        file_path = available_files[selected_dataset]
        st.info(f"Loading: {os.path.basename(file_path)}")
        
        # Load data with proper handling
        if 'k2pandc' in file_path:
            df = pd.read_csv(file_path, comment='#')
        else:
            df = pd.read_csv(file_path, comment='#')
        
        return df, selected_dataset
    except FileNotFoundError:
        st.error(f"Data file not found: {file_path}")
        return None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def plot_light_curve(df, star_id):
    """Plot light curve for a specific star"""
    star_data = df[df['id'] == star_id]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=star_data['time'],
        y=star_data['flux'],
        mode='lines+markers',
        name=f'Star {star_id}',
        line=dict(width=2)
    ))
    
    fig.update_layout(
        title=f'Light Curve for Star {star_id}',
        xaxis_title='Time',
        yaxis_title='Normalized Flux',
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_model_results(y_test, y_pred, y_pred_proba):
    """Plot model evaluation results"""
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Confusion Matrix', 'Prediction Probability Distribution'),
        specs=[[{"type": "heatmap"}, {"type": "histogram"}]]
    )
    
    # Confusion matrix heatmap
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=['No Exoplanet', 'Exoplanet'],
            y=['No Exoplanet', 'Exoplanet'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16}
        ),
        row=1, col=1
    )
    
    # Probability distribution
    fig.add_trace(
        go.Histogram(
            x=y_pred_proba,
            nbinsx=20,
            name='Prediction Probabilities',
            opacity=0.7
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Model Evaluation Results",
        template='plotly_white',
        height=400
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="NASA Exoplanet Detection Challenge 2025",
        page_icon="🌌",
        layout="wide"
    )
    
    st.title("🌌 NASA Exoplanet Detection Challenge 2025")
    st.markdown("**Advanced Machine Learning Interface for Exoplanet Discovery**")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Data Input", "Model Training", "Results Analysis", "Real Data Explorer", "LSTM Deep Learning"]
    )
    
    if page == "Data Input":
        st.header("📊 Data Input & Configuration")
        
        data_source = st.radio(
            "Select data source:",
            ["Generate Synthetic Data", "Upload CSV File", "Use Real Data"]
        )
        
        if data_source == "Generate Synthetic Data":
            st.subheader("Synthetic Data Parameters")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                num_stars = st.slider("Number of stars", 100, 2000, 1000)
            with col2:
                time_points = st.slider("Time points per star", 100, 1000, 500)
            with col3:
                exoplanet_ratio = st.slider("Exoplanet ratio", 0.1, 0.9, 0.5)
            
            if st.button("Generate Data", type="primary"):
                with st.spinner("Generating synthetic data..."):
                    df, y = generate_synthetic_data(num_stars, time_points, exoplanet_ratio)
                    st.session_state.df = df
                    st.session_state.y = y
                    
                st.success(f"Generated data for {num_stars} stars with {time_points} time points each")
                
                # Show data preview
                st.subheader("Data Preview")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Light curve data shape:", df.shape)
                    st.dataframe(df.head(10))
                
                with col2:
                    st.write("Target distribution:")
                    target_counts = y.value_counts()
                    st.write(f"No exoplanet: {target_counts[0]}")
                    st.write(f"Exoplanet: {target_counts[1]}")
                    
                    # Plot sample light curve
                    if st.button("Show sample light curve"):
                        sample_star = np.random.choice(df['id'].unique())
                        fig = plot_light_curve(df, sample_star)
                        st.plotly_chart(fig, use_container_width=True)
        
        elif data_source == "Upload CSV File":
            st.subheader("Upload Your Data")
            uploaded_file = st.file_uploader(
                "Choose a CSV file", 
                type="csv",
                help="File should have columns: id, time, flux"
            )
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_df = df
                
                st.write("Uploaded data shape:", df.shape)
                st.dataframe(df.head())
                
                # Allow user to specify target column
                if 'target' not in df.columns:
                    st.warning("No 'target' column found. Please specify labels manually or use synthetic labels.")
                    if st.button("Generate random labels for demonstration"):
                        y = pd.Series(np.random.choice([0, 1], size=df['id'].nunique()))
                        st.session_state.y = y
        
        elif data_source == "Use Real Data":
            st.subheader("Real Exoplanet Data")
            result = load_real_data()
            if result is not None and len(result) == 2:
                real_data, dataset_name = result
                if real_data is not None:
                    st.success(f"Loaded {dataset_name} successfully!")
                    st.write("Data shape:", real_data.shape)
                    st.dataframe(real_data.head())
                    st.session_state.real_data = real_data
                    st.session_state.dataset_name = dataset_name
    
    elif page == "Model Training":
        st.header("🤖 Model Training")
        
        # Check if we have data from any source
        has_synthetic_data = 'df' in st.session_state and 'y' in st.session_state
        has_real_data = 'real_data' in st.session_state
        has_uploaded_data = 'uploaded_df' in st.session_state
        
        if has_synthetic_data:
            st.success("✅ Synthetic data loaded")
            df = st.session_state.df
            y = st.session_state.y
            data_source = "Synthetic Data"
            
        elif has_real_data:
            st.info("ℹ️ Real exoplanet data detected - converting for model training")
            
            # Real data needs to be converted to time series format for training
            #st.warning("⚠️ Real data contains planetary parameters, not light curves. Consider using synthetic data for time-series ML training.")
            
            # Option 1: Generate synthetic data based on real data parameters
            real_data = st.session_state.real_data
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Real data info:")
                st.write(f"Records: {len(real_data):,}")
                if 'disposition' in real_data.columns:
                    confirmed = len(real_data[real_data['disposition'] == 'CONFIRMED'])
                    st.write(f"Confirmed exoplanets: {confirmed:,}")
            
            with col2:
                st.write("Training options:")
                training_option = st.radio(
                    "Choose training approach:",
                    [
                        "Generate synthetic data for ML training",
                        "Use real data parameters (limited ML applicability)"
                    ]
                )
            
            if training_option == "Generate synthetic data for ML training":
                if st.button("Generate Training Data from Real Parameters", type="primary"):
                    # Use real data to inform synthetic generation
                    num_confirmed = len(real_data[real_data['disposition'] == 'CONFIRMED']) if 'disposition' in real_data.columns else 100
                    num_candidates = len(real_data[real_data['disposition'] == 'CANDIDATE']) if 'disposition' in real_data.columns else 100
                    
                    total_stars = min(1000, num_confirmed + num_candidates)
                    exoplanet_ratio = num_confirmed / (num_confirmed + num_candidates) if (num_confirmed + num_candidates) > 0 else 0.5
                    
                    with st.spinner("Generating synthetic data based on real parameters..."):
                        df, y = generate_synthetic_data(total_stars, 500, exoplanet_ratio)
                        st.session_state.df = df
                        st.session_state.y = y
                    
                    st.success(f"Generated {total_stars} synthetic light curves with {exoplanet_ratio:.2%} exoplanet ratio based on real data")
                    data_source = "Synthetic (based on real parameters)"
                else:
                    st.info("👆 Click the button above to generate training data")
                    return
            else:
                st.error("❌ Direct ML training on real data parameters is not yet implemented. Please use synthetic data generation.")
                return
                
        elif has_uploaded_data:
            uploaded_df = st.session_state.uploaded_df
            st.info("📁 Uploaded data detected")
            
            # Check if uploaded data has the right format
            required_cols = ['id', 'time', 'flux']
            missing_cols = [col for col in required_cols if col not in uploaded_df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.write("Required columns: id, time, flux")
                st.write(f"Available columns: {list(uploaded_df.columns)}")
                return
            else:
                df = uploaded_df
                # Need to generate or specify target labels
                if 'y' not in st.session_state:
                    st.warning("No target labels found. Generating random labels for demonstration.")
                    unique_ids = df['id'].nunique()
                    y = pd.Series(np.random.choice([0, 1], size=unique_ids))
                    st.session_state.y = y
                else:
                    y = st.session_state.y
                data_source = "Uploaded Data"
        else:
            st.warning("⚠️ No data loaded! Please go to the Data Input page first.")
            st.info("💡 Available options:")
            st.write("• Generate Synthetic Data")
            st.write("• Upload CSV File")
            st.write("• Use Real K2 Data (will generate synthetic training data)")
            return
        
        # Display current data info
        st.subheader(f"📊 Current Data: {data_source}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Data Source", data_source)
        with col2:
            st.metric("Total Stars", len(y))
        with col3:
            st.metric("Exoplanets", y.sum())
        with col4:
            if hasattr(df, 'shape'):
                time_points = len(df[df['id'] == df['id'].iloc[0]]) if len(df) > 0 else 0
                st.metric("Time Points", time_points)
        
        # Training section
        st.subheader("🚀 Start Training")
            
        # Training section
        st.subheader("🚀 Start Training")
        
        if st.button("Start Training", type="primary"):
            model, X, y_data, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, accuracy = train_tsfresh_model(df, y)
            
            # Store results in session state
            st.session_state.model = model
            st.session_state.X = X
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.y_pred_proba = y_pred_proba
            st.session_state.accuracy = accuracy
            st.session_state.data_source = data_source
            
            st.success(f"Model trained successfully! Accuracy: {accuracy:.4f}")
            
            # Quick results
            st.subheader("Quick Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
            with col2:
                st.metric("Features Used", X.shape[1])
            with col3:
                st.metric("Test Samples", len(y_test))
    
    elif page == "Results Analysis":
        st.header("📈 Results Analysis")
        
        if 'model' in st.session_state:
            accuracy = st.session_state.accuracy
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred
            y_pred_proba = st.session_state.y_pred_proba
            
            # Overall metrics
            st.subheader("Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
            with col2:
                precision = ((y_pred == 1) & (y_test == 1)).sum() / (y_pred == 1).sum() if (y_pred == 1).sum() > 0 else 0
                st.metric("Precision", f"{precision:.4f}")
            with col3:
                recall = ((y_pred == 1) & (y_test == 1)).sum() / (y_test == 1).sum() if (y_test == 1).sum() > 0 else 0
                st.metric("Recall", f"{recall:.4f}")
            with col4:
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                st.metric("F1-Score", f"{f1:.4f}")
            
            # Visualization
            st.subheader("Model Evaluation Visualizations")
            fig = plot_model_results(y_test, y_pred, y_pred_proba)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            if hasattr(st.session_state.model, 'feature_importances_'):
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'feature': st.session_state.X.columns,
                    'importance': st.session_state.model.feature_importances_
                }).sort_values('importance', ascending=False).head(20)
                
                fig = px.bar(
                    feature_importance, 
                    x='importance', 
                    y='feature', 
                    orientation='h',
                    title='Top 20 Most Important Features'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Classification report
            st.subheader("Detailed Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            # Model export
            st.subheader("Export Model")
            if st.button("Save Model"):
                joblib.dump(st.session_state.model, 'exoplanet_model.pkl')
                st.success("Model saved as 'exoplanet_model.pkl'")
        else:
            st.warning("Please train a model first!")
    
    elif page == "LSTM Deep Learning":
        st.header("🧠 LSTM Deep Learning")
        
        if not LSTM_AVAILABLE:
            st.error("❌ LSTM functionality not available")
            st.info("Install required packages: `pip install tensorflow astropy`")
            return
        
        st.info("🌟 Advanced deep learning approach using LSTM neural networks on real Kepler light curve data")
        
        # LSTM Configuration
        st.subheader("⚙️ LSTM Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            lstm_epochs = st.slider("Training Epochs", 5, 100, 20, 
                                  help="Number of training epochs (more = better training, longer time)")
            lstm_batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1,
                                         help="Training batch size")
            lstm_model_type = st.selectbox("Model Architecture", 
                                         ["simple", "advanced"], 
                                         help="Simple: faster, Advanced: more powerful")
        
        with col2:
            sequence_length = st.slider("Sequence Length", 100, 2000, 1000,
                                      help="Length of light curve sequences for LSTM")
            use_real_data = st.checkbox("Use Real Kepler FITS Data", value=True,
                                      help="Use actual Kepler telescope data")
            save_model = st.checkbox("Save Trained Model", value=True)
        
        # Data source info
        if use_real_data:
            st.info("📡 Will use real Kepler FITS files from data/ directory")
            
            # Check for available data
            data_dirs = ['data/Kepler_confirmed_wget', 'data/Kepler_KOI_wget']
            available_data = []
            for data_dir in data_dirs:
                if os.path.exists(data_dir):
                    fits_files = len([f for f in os.listdir(data_dir) if f.endswith('.fits')])
                    if fits_files > 0:
                        available_data.append(f"{data_dir}: {fits_files} FITS files")
            
            if available_data:
                st.success("✅ Available Kepler data:")
                for data_info in available_data:
                    st.write(f"  • {data_info}")
            else:
                st.warning("⚠️ No FITS files found in data directories")
                st.write("Expected directories: data/Kepler_confirmed_wget/, data/Kepler_KOI_wget/")
        else:
            st.info("🔧 Will generate synthetic light curve data for training")
        
        # Training section
        st.subheader("🚀 LSTM Training")
        
        lstm_config = {
            'epochs': lstm_epochs,
            'batch_size': lstm_batch_size,
            'model_type': lstm_model_type,
            'sequence_length': sequence_length,
            'save_model': save_model,
            'data_root': 'data' if use_real_data else None
        }
        
        if st.button("🧠 Start LSTM Training", type="primary"):
            with st.spinner("Training LSTM model..."):
                lstm_results = train_lstm_model(lstm_config)
            
            if lstm_results:
                st.success("🎉 LSTM training completed!")
                
                # Store results
                st.session_state.lstm_results = lstm_results
                st.session_state.lstm_model_type = 'lstm'
                
                # Display quick results
                results = lstm_results['results']
                
                if 'evaluation' in results:
                    eval_results = results['evaluation']
                    
                    st.subheader("📊 LSTM Performance")
                    
                    # Get test results
                    if 'test_default_threshold' in eval_results:
                        test_metrics = eval_results['test_default_threshold']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{test_metrics['accuracy']:.4f}")
                        with col2:
                            st.metric("Precision", f"{test_metrics['precision']:.4f}")
                        with col3:
                            st.metric("Recall", f"{test_metrics['recall']:.4f}")
                        with col4:
                            st.metric("F1-Score", f"{test_metrics['f1_score']:.4f}")
                        
                        # ROC AUC
                        st.metric("ROC AUC", f"{test_metrics['roc_auc']:.4f}")
                    
                    # Data info
                    if 'data_summary' in results:
                        data_info = results['data_summary']
                        st.subheader("📈 Training Data Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Light Curves", data_info['total_light_curves'])
                        with col2:
                            st.metric("Confirmed Exoplanets", data_info['positive_class_count'])
                        with col3:
                            st.metric("Non-Exoplanets", data_info['negative_class_count'])
                
                # Model architecture info
                pipeline = lstm_results['pipeline']
                if pipeline.model:
                    st.subheader("🏗️ Model Architecture")
                    st.text(pipeline.model.get_model_summary())
                
                st.info("🔍 Go to 'Results Analysis' page to see detailed LSTM results and comparisons!")
            else:
                st.error("❌ LSTM training failed. Check the logs above for details.")
        
        # Quick comparison if both models exist
        if 'model' in st.session_state and 'lstm_results' in st.session_state:
            st.subheader("⚔️ Quick Model Comparison")
            
            # Get tsfresh accuracy
            tsfresh_acc = st.session_state.accuracy if 'accuracy' in st.session_state else 0
            
            # Get LSTM accuracy
            lstm_acc = 0
            if 'evaluation' in st.session_state.lstm_results['results']:
                eval_results = st.session_state.lstm_results['results']['evaluation']
                if 'test_default_threshold' in eval_results:
                    lstm_acc = eval_results['test_default_threshold']['accuracy']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("tsfresh + LightGBM", f"{tsfresh_acc:.4f}")
            with col2:
                st.metric("LSTM Neural Network", f"{lstm_acc:.4f}")
            
            if lstm_acc > tsfresh_acc:
                st.success("🏆 LSTM is performing better!")
            elif tsfresh_acc > lstm_acc:
                st.success("🏆 tsfresh + LightGBM is performing better!")
            else:
                st.info("🤝 Both models have similar performance!")
    
    elif page == "Real Data Explorer":
        st.header("🔍 Real K2 Data Explorer")
        
        if 'real_data' in st.session_state:
            df = st.session_state.real_data
            
            st.subheader("Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                confirmed = len(df[df['disposition'] == 'CONFIRMED'])
                st.metric("Confirmed Exoplanets", confirmed)
            with col3:
                candidates = len(df[df['disposition'] == 'CANDIDATE'])
                st.metric("Candidates", candidates)
            
            # Interactive filters
            st.subheader("Data Filters")
            col1, col2 = st.columns(2)
            
            with col1:
                disposition_filter = st.multiselect(
                    "Filter by disposition:",
                    df['disposition'].unique(),
                    default=df['disposition'].unique()
                )
            
            with col2:
                discovery_method = st.multiselect(
                    "Filter by discovery method:",
                    df['discoverymethod'].unique(),
                    default=df['discoverymethod'].unique()
                )
            
            # Apply filters
            filtered_df = df[
                (df['disposition'].isin(disposition_filter)) &
                (df['discoverymethod'].isin(discovery_method))
            ]
            
            st.write(f"Showing {len(filtered_df)} records")
            st.dataframe(filtered_df.head(20))
            
            # Visualizations
            st.subheader("Data Visualizations")
            
            # Disposition distribution
            if len(filtered_df) > 0:
                fig1 = px.pie(
                    filtered_df, 
                    names='disposition', 
                    title='Distribution of Exoplanet Status'
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Planet radius distribution
                if 'pl_rade' in filtered_df.columns:
                    fig2 = px.histogram(
                        filtered_df.dropna(subset=['pl_rade']),
                        x='pl_rade',
                        nbins=30,
                        title='Distribution of Planet Radius (Earth Radii)',
                        labels={'pl_rade': 'Planet Radius (Earth Radii)'}
                    )
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Please load real data first from the Data Input page!")

if __name__ == "__main__":
    main()