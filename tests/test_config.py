"""Test configuration and utilities."""
import unittest
import json
from pathlib import Path
import os

from src.utils.config import load_config, get_model_config, validate_config

class TestConfigUtilities(unittest.TestCase):
    def setUp(self):
        """Prepare test environment."""
        self.config_dir = Path('config')
        self.test_configs = ['lnt', 'wcst']  # Add more configs as needed

    def test_config_file_existence(self):
        """Verify that all expected config files exist."""
        for config_name in self.test_configs:
            config_path = self.config_dir / f'{config_name}_config.json'
            self.assertTrue(config_path.exists(), 
                            f"Config file for {config_name} does not exist")

    def test_config_loading(self):
        """Test loading of configuration files."""
        for config_name in self.test_configs:
            try:
                config = load_config(config_name)
                
                # Verify basic config structure
                self.assertIsInstance(config, dict)
                
                # Verify required top-level keys
                required_keys = ['standard_test', 'component_tasks', 'model_configs']
                for key in required_keys:
                    self.assertIn(key, config, 
                                f"Missing {key} in {config_name} configuration")
            except Exception as e:
                self.fail(f"Failed to load {config_name} configuration: {e}")

    def test_model_config_extraction(self):
        """Test extraction of model-specific configurations."""
        test_model_cases = [
            ('gpt-3.5-turbo', 'openai'),
            ('gpt-4', 'openai'),
            ('gemini-1.5-pro', 'gemini'),
            ('llama-70b', 'llama')
        ]
        
        for config_name in self.test_configs:
            config = load_config(config_name)
            
            for model_name, expected_type in test_model_cases:
                try:
                    model_config = get_model_config(config, model_name)
                    
                    # Verify basic model config structure
                    self.assertIsInstance(model_config, dict)
                    
                    # Verify temperature setting exists
                    self.assertIn('temperature', model_config)
                    
                    # Verify temperature is a float between 0 and 1
                    self.assertIsInstance(model_config['temperature'], float)
                    self.assertTrue(0 <= model_config['temperature'] <= 1)
                except Exception as e:
                    self.fail(f"Failed to extract config for {model_name}: {e}")

    def test_config_validation(self):
        """Test configuration validation."""
        for config_name in self.test_configs:
            config = load_config(config_name)
            
            try:
                validate_config(config)
            except ValueError as e:
                self.fail(f"Validation failed for {config_name} config: {e}")
            
            # Test invalid config scenarios
            invalid_configs = [
                {},  # Empty config
                {'standard_test': {}},  # Partial config
                {'standard_test': {'models': []}}  # Empty models list
            ]
            
            for invalid_config in invalid_configs:
                with self.assertRaises(ValueError, 
                    msg=f"Failed to catch invalid config for {config_name}"):
                    validate_config(invalid_config)

    def test_model_list_in_config(self):
        """Verify that model lists contain expected models."""
        expected_models = {
            'gpt-3.5-turbo', 'gpt-4', 
            'gemini-1.5-pro', 'llama-70b'
        }
        
        for config_name in self.test_configs:
            config = load_config(config_name)
            
            # Check models in standard test
            standard_models = set(config.get('standard_test', {}).get('models', []))
            self.assertTrue(standard_models.issubset(expected_models), 
                            f"Unexpected models in {config_name} standard test")
            
            # Check models in component tasks
            component_models = set(config.get('component_tasks', {}).get('models', []))
            self.assertTrue(component_models.issubset(expected_models), 
                            f"Unexpected models in {config_name} component tasks")

def run_config_tests():
    """Run all configuration tests."""
    unittest.main()

if __name__ == '__main__':
    run_config_tests()