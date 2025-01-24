"""Configuration loading utilities."""
import json
from pathlib import Path
from typing import Dict, Any

def load_config(test_type: str) -> Dict[str, Any]:
    """Load configuration for specified test type."""
    config_path = Path('config')
    config_file = config_path / f'{test_type}_config.json'
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
        
    with open(config_file) as f:
        return json.load(f)

def get_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Get model-specific configuration."""
    if model_name.startswith('gpt'):
        return config['model_configs']['openai']
    elif model_name.startswith('gemini'):
        return config['model_configs']['gemini']
    elif model_name.startswith('llama'):
        return config['model_configs']['llama']
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration file structure."""
    required_keys = ['standard_test', 'component_tasks', 'model_configs']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key: {key}")
            
    if not config['standard_test'].get('models'):
        raise ValueError("No models specified in standard_test configuration")