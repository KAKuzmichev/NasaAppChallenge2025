"""
Configuration loader for LightGBM parameters
Loads YAML configuration and provides easy access to model parameters
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

class LGBMConfigLoader:
    """
    Loader class for LightGBM configuration from YAML file
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the config loader
        
        Args:
            config_path: Path to the YAML config file. If None, uses default path.
        """
        if config_path is None:
            # Default path relative to this file
            current_dir = Path(__file__).parent
            config_path = str(current_dir / "LGBMClassifier_Config.yaml")
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default LightGBM parameters (combines all parameter groups)
        
        Returns:
            Dictionary with all default parameters
        """
        params = {}
        
        # Combine all parameter groups
        for group_name in ['core_parameters', 'tree_parameters', 
                          'sampling_regularization', 'performance_optimization']:
            if group_name in self.config:
                params.update(self.config[group_name])
        
        return params
    
    def get_config_params(self, config_name: str) -> Dict[str, Any]:
        """
        Get parameters for a specific configuration
        
        Args:
            config_name: Name of the configuration (e.g., 'fast_training', 'high_accuracy')
            
        Returns:
            Dictionary with configuration parameters
        """
        if 'configurations' not in self.config:
            raise ValueError("No configurations found in config file")
        
        if config_name not in self.config['configurations']:
            available = list(self.config['configurations'].keys())
            raise ValueError(f"Configuration '{config_name}' not found. Available: {available}")
        
        # Start with default parameters
        params = self.get_default_params()
        
        # Override with specific configuration
        config_params = self.config['configurations'][config_name]
        params.update(config_params)
        
        return params
    
    def get_parameter_group(self, group_name: str) -> Dict[str, Any]:
        """
        Get parameters from a specific group
        
        Args:
            group_name: Name of the parameter group
            
        Returns:
            Dictionary with group parameters
        """
        if group_name not in self.config:
            available = [k for k in self.config.keys() if not k.startswith('_')]
            raise ValueError(f"Group '{group_name}' not found. Available: {available}")
        
        return self.config[group_name].copy()
    
    def list_configurations(self) -> list:
        """
        List all available configurations
        
        Returns:
            List of configuration names
        """
        if 'configurations' in self.config:
            return list(self.config['configurations'].keys())
        return []
    
    def print_config_summary(self):
        """Print a summary of the configuration"""
        print("🔧 LightGBM Configuration Summary")
        print("=" * 40)
        
        # Default parameters
        default_params = self.get_default_params()
        print(f"\n📊 Default Parameters ({len(default_params)} total):")
        for key, value in default_params.items():
            print(f"   {key}: {value}")
        
        # Available configurations
        configs = self.list_configurations()
        if configs:
            print(f"\n⚙️  Available Configurations ({len(configs)} total):")
            for config in configs:
                print(f"   • {config}")
    
    def export_to_python(self, output_path: str, config_name: Optional[str] = None):
        """
        Export configuration to Python format
        
        Args:
            output_path: Path to save the Python file
            config_name: Specific configuration to export (if None, exports default)
        """
        if config_name:
            params = self.get_config_params(config_name)
            file_comment = f"# LightGBM Parameters - {config_name} configuration"
        else:
            params = self.get_default_params()
            file_comment = "# LightGBM Parameters - Default configuration"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(file_comment + "\n")
            f.write("# Generated from YAML configuration\n\n")
            f.write("LGBM_PARAMS = {\n")
            for key, value in params.items():
                if isinstance(value, str):
                    f.write(f"    '{key}': '{value}',\n")
                else:
                    f.write(f"    '{key}': {value},\n")
            f.write("}\n")

def load_lgbm_config(config_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to load LightGBM configuration
    
    Args:
        config_name: Name of specific configuration to load (if None, loads default)
        
    Returns:
        Dictionary with LightGBM parameters
    """
    loader = LGBMConfigLoader()
    
    if config_name:
        return loader.get_config_params(config_name)
    else:
        return loader.get_default_params()

if __name__ == "__main__":
    # Demo usage
    loader = LGBMConfigLoader()
    loader.print_config_summary()
    
    print("\n🧪 Testing configuration loading:")
    
    # Test default config
    default_params = loader.get_default_params()
    print(f"   Default config loaded: {len(default_params)} parameters")
    
    # Test specific config
    try:
        fast_params = loader.get_config_params('fast_training')
        print(f"   Fast training config loaded: {len(fast_params)} parameters")
    except ValueError as e:
        print(f"   Error: {e}")
    
    # Export example
    loader.export_to_python("lgbm_params_export.py", "high_accuracy")
    print("   Exported high_accuracy config to Python file")