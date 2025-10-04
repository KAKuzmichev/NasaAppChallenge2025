"""
LightGBM Configuration Manager
Provides different configurations for LightGBM without external YAML dependency
"""

def get_lgbm_config(config_name: str = 'default'):
    """
    Get LightGBM configuration parameters
    
    Args:
        config_name: Configuration name ('default', 'fast', 'accurate', 'gpu', 'large_data')
        
    Returns:
        Dictionary with LightGBM parameters
    """
    
    # Base configuration (translated from original config)
    base_config = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'binary_logloss',
        'seed': 123,
        'verbose': -1,  # Suppress LightGBM warnings
        'random_state': 123
    }
    
    configurations = {
        'default': {
            **base_config,
            'n_estimators': 50,
            'learning_rate': 0.018,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 25,
            'min_child_weight': 0.0001,
            'max_bin': 255,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'min_gain_to_split': 0.0,
            'n_jobs': -1,
            'device': 'cpu',
            'tree_learner': 'data',
            'scale_pos_weight': 1.0
        },
        
        'fast': {
            **base_config,
            'n_estimators': 30,
            'learning_rate': 0.1,
            'num_leaves': 15,
            'min_child_samples': 50,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'n_jobs': -1
        },
        
        'accurate': {
            **base_config,
            'n_estimators': 200,
            'learning_rate': 0.01,
            'num_leaves': 63,
            'min_child_samples': 10,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 1,
            'lambda_l1': 0.05,
            'lambda_l2': 0.05,
            'n_jobs': -1
        },
        
        'gpu': {
            **base_config,
            'device': 'gpu',
            'tree_learner': 'data',
            'n_estimators': 100,
            'learning_rate': 0.05,
            'num_leaves': 255,
            'max_bin': 1023,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8
        },
        
        'large_data': {
            **base_config,
            'n_estimators': 100,
            'learning_rate': 0.05,
            'num_leaves': 127,
            'max_bin': 512,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'min_child_samples': 50,
            'n_jobs': -1
        }
    }
    
    if config_name not in configurations:
        available = list(configurations.keys())
        raise ValueError(f"Configuration '{config_name}' not found. Available: {available}")
    
    return configurations[config_name]

def list_available_configs():
    """List all available configuration names"""
    return ['default', 'fast', 'accurate', 'gpu', 'large_data']

def get_config_description(config_name: str):
    """Get description of a configuration"""
    descriptions = {
        'default': 'Balanced configuration based on original parameters',
        'fast': 'Fast training with reduced complexity (less accuracy)',
        'accurate': 'High accuracy with more iterations and fine-tuning',
        'gpu': 'GPU-optimized configuration for faster training',
        'large_data': 'Optimized for large datasets with memory efficiency'
    }
    return descriptions.get(config_name, 'No description available')

def print_config_info():
    """Print information about all available configurations"""
    print("🔧 Available LightGBM Configurations:")
    print("=" * 50)
    
    for config_name in list_available_configs():
        desc = get_config_description(config_name)
        params = get_lgbm_config(config_name)
        
        print(f"\n📊 {config_name.upper()}:")
        print(f"   Description: {desc}")
        print(f"   Key params: n_estimators={params['n_estimators']}, "
              f"learning_rate={params['learning_rate']}, "
              f"num_leaves={params['num_leaves']}")

if __name__ == "__main__":
    print_config_info()
    
    # Test loading a configuration
    print(f"\n🧪 Testing default configuration:")
    config = get_lgbm_config('default')
    print(f"   Loaded {len(config)} parameters")
    
    # Show some key parameters
    key_params = ['objective', 'n_estimators', 'learning_rate', 'num_leaves']
    for param in key_params:
        print(f"   {param}: {config[param]}")