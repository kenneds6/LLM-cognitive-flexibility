"""Integration tests for cognitive flexibility experiments."""
import unittest
import tempfile
import json
import os
from pathlib import Path
import pandas as pd
import random

from src.tests.wcst import WCST, WCSTConfig
from src.tests.lnt import LNT, LNTConfig
from src.models.openai import OpenAIWrapper, OpenAIConfig
from src.utils.config import load_config, get_model_config, validate_config
from src.utils.logging import setup_logger

class MockModel:
    def __init__(self):
        self.call_count = 0
        self.conversation_history = []
        
    def send_message(self, message: str, system_prompt: str = "") -> str:
        """
        Mock send_message method that provides deterministic responses
        for different test scenarios.
        """
        self.call_count += 1
        
        # Deterministic responses based on the test
        if "WCST" in system_prompt:
            return "1"  # Always choose first option
        elif "LNT" in system_prompt:
            return "vowel"  # Always choose vowel for LNT
        
        return ""
        
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        self.call_count = 0
        
    def _extract_choice(self, response: str) -> int:
        """Mock choice extraction method."""
        return 0
    
    def _extract_ln_response(self, response: str) -> str:
        """Mock letter-number response extraction."""
        return "vowel"

class TestExperimentComponents(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.results_dir = Path(self.temp_dir) / "results"
        self.results_dir.mkdir(exist_ok=True)

    def test_wcst_initialization(self):
        """Test WCST initialization and basic functionality."""
        # Test default configuration
        wcst_default = WCST()
        self.assertEqual(len(wcst_default.deck), 5 * 4 * 4 * 4)  # Total cards
        self.assertIn(wcst_default.current_rule, ['shape', 'color', 'number'])
        
        # Test custom configuration
        custom_config = WCSTConfig(
            num_trials=10, 
            num_successes_before_switch=3,
            shapes=['circle', 'triangle'],
            colors=['red', 'blue'],
            numbers=[1, 2]
        )
        wcst_custom = WCST(custom_config)
        self.assertEqual(len(wcst_custom.deck), 5 * 2 * 2 * 2)
        
    def test_wcst_option_generation(self):
        """Test WCST option generation."""
        wcst = WCST()
        card = wcst.deck[0]
        options = wcst.generate_options(card)
        
        # Verify options
        self.assertEqual(len(options), 4)
        
        # Verify each option is a valid card
        for option in options:
            self.assertEqual(len(option), 3)
            self.assertIn(option[0], wcst.config.shapes)
            self.assertIn(option[1], wcst.config.colors)
            self.assertIn(option[2], wcst.config.numbers)
        
    def test_wcst_rule_switching(self):
        """Test rule switching mechanism."""
        wcst = WCST()
        initial_rule = wcst.current_rule
        other_rules = [rule for rule in ['shape', 'color', 'number'] if rule != initial_rule]
        
        # Simulate 6 successful trials
        for i in range(6):
            # Always choose the first option that matches the rule
            options = wcst.generate_options(wcst.deck[i])
            matching_option_index = next(
                (j for j, opt in enumerate(options) 
                 if opt[{'shape': 0, 'color': 1, 'number': 2}[initial_rule]] == 
                    wcst.deck[i][{'shape': 0, 'color': 1, 'number': 2}[initial_rule]]), 
                0
            )
            
            wcst.evaluate_choice(wcst.deck[i], matching_option_index, options)
        
        # Rule should now be one of the other rules
        self.assertIn(wcst.current_rule, other_rules)
        
    def test_lnt_initialization(self):
        """Test LNT initialization and basic functionality."""
        # Test default configuration
        lnt_default = LNT()
        self.assertIn(lnt_default.current_task, ['letter', 'number'])
        self.assertEqual(lnt_default.score, 0)
        
        # Test custom configuration
        custom_config = LNTConfig(num_trials=10, num_successes_before_switch=3)
        lnt_custom = LNT(custom_config)
        self.assertEqual(lnt_custom.config.num_trials, 10)
        
    def test_lnt_sequence_generation(self):
        """Test Letter Number Test sequence generation."""
        lnt = LNT()
        sequence = lnt.generate_sequence()
        
        # Verify sequence format
        self.assertEqual(len(sequence), 2)
        self.assertTrue(sequence[0].isalpha())
        self.assertTrue(sequence[1].isdigit())
        
    def test_lnt_response_evaluation(self):
        """Test LNT response evaluation."""
        lnt = LNT()
        
        # Test vowel task
        lnt.current_task = 'letter'
        
        # Vowel letter test
        vowel_sequence = 'a5'
        self.assertTrue(lnt.evaluate_response(vowel_sequence, 'vowel'))
        
        # Consonant letter test
        consonant_sequence = 'b5'
        self.assertTrue(lnt.evaluate_response(consonant_sequence, 'consonant'))
        
        # Number task
        lnt.current_task = 'number'
        
        # Even number test
        even_sequence = 'x2'
        self.assertTrue(lnt.evaluate_response(even_sequence, 'even'))
        
        # Odd number test
        odd_sequence = 'x3'
        self.assertTrue(lnt.evaluate_response(odd_sequence, 'odd'))

    def test_config_loading(self):
        """Test configuration loading utilities."""
        # Load LNT config
        lnt_config = load_config('lnt')
        
        # Verify config structure
        self.assertIn('standard_test', lnt_config)
        self.assertIn('component_tasks', lnt_config)
        self.assertIn('model_configs', lnt_config)
        
        # Test model config extraction
        openai_config = get_model_config(lnt_config, 'gpt-3.5-turbo')
        self.assertIn('temperature', openai_config)
        
        # Test config validation
        try:
            validate_config(lnt_config)
        except ValueError as e:
            self.fail(f"Config validation failed: {e}")
        
    def test_logging_setup(self):
        """Test logger setup."""
        logger = setup_logger('test_model', 'test_task')
        
        # Verify logger properties
        self.assertTrue(hasattr(logger, 'info'))
        self.assertTrue(hasattr(logger, 'error'))
        
        # Check log file creation
        log_dir = Path('logs')
        log_files = list(log_dir.glob(f"test_task_test_model_*.log"))
        self.assertTrue(len(log_files) > 0)
        
    def test_mock_model(self):
        """Test mock model functionality."""
        mock_model = MockModel()
        
        # Test send_message
        response = mock_model.send_message("Test message", "WCST system prompt")
        self.assertEqual(response, "1")
        self.assertEqual(mock_model.call_count, 1)
        
        # Test reset
        mock_model.reset_conversation()
        self.assertEqual(mock_model.call_count, 0)

def run_integration_tests():
    """Run all integration tests."""
    unittest.main()

