"""
Tests for analysis and visualization components of cognitive flexibility experiments.
"""
import unittest
import tempfile
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from src.analysis.analyze_results import (
        load_results, 
        calculate_statistics, 
        calculate_bounds, 
        plot_accuracy_distribution,
        analyze_all
    )
    from src.analysis.component_analysis import analyze_component_tasks
except ImportError:
    # Fallback for development/testing
    from analysis.analyze_results import (
        load_results, 
        calculate_statistics, 
        calculate_bounds, 
        plot_accuracy_distribution,
        analyze_all
    )
    from analysis.component_analysis import analyze_component_tasks

class TestAnalysisComponents(unittest.TestCase):
    def setUp(self):
        """Set up test environment with temporary directories."""
        self.temp_dir = tempfile.mkdtemp()
        self.results_dir = os.path.join(self.temp_dir, 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        self.output_dir = os.path.join(self.temp_dir, 'analysis_output')
        os.makedirs(self.output_dir, exist_ok=True)

    def _create_mock_results(self):
        """Create mock results files for testing."""
        # Mock WCST results
        wcst_results = [
            {"evaluation": 1, "accuracy": 0.8, "score": 20, "trials": 25, "model": "gpt-3.5-turbo"},
            {"evaluation": 2, "accuracy": 0.75, "score": 18, "trials": 25, "model": "gpt-3.5-turbo"},
            {"evaluation": 1, "accuracy": 0.9, "score": 22, "trials": 25, "model": "gpt-4"},
        ]
        
        # Mock LNT results
        lnt_results = [
            {"evaluation": 1, "accuracy": 0.7, "score": 17, "trials": 25, "model": "gpt-3.5-turbo"},
            {"evaluation": 2, "accuracy": 0.65, "score": 16, "trials": 25, "model": "gpt-3.5-turbo"},
            {"evaluation": 1, "accuracy": 0.85, "score": 21, "trials": 25, "model": "gpt-4"},
        ]
        
        # Save mock results files
        wcst_file = os.path.join(self.results_dir, 'wcst_gpt-3.5-turbo_mock.json')
        lnt_file = os.path.join(self.results_dir, 'lnt_gpt-3.5-turbo_mock.json')
        
        with open(wcst_file, 'w') as f:
            json.dump(wcst_results, f)
        
        with open(lnt_file, 'w') as f:
            json.dump(lnt_results, f)

    def test_results_loading(self):
        """Test loading results from JSON files."""
        # Create mock result files
        self._create_mock_results()
        
        # Load results
        results = load_results(self.results_dir)
        
        # Check loaded data
        self.assertIn('wcst', results)
        self.assertIn('lnt', results)
        
        # Verify DataFrame properties
        for test_name, df in results.items():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertIn('model', df.columns)
            self.assertIn('accuracy', df.columns)

    def test_statistics_calculation(self):
        """Test calculation of performance statistics."""
        # Create mock result files
        self._create_mock_results()
        
        # Load results
        results = load_results(self.results_dir)
        
        # Calculate statistics for each test
        for test_name, df in results.items():
            stats = calculate_statistics(df)
            
            # Check statistical columns
            expected_columns = [
                'mean_acc', 'std_acc', 'min_acc', 'max_acc', 
                'mean_score', 'std_score', 'avg_trials'
            ]
            for col in expected_columns:
                self.assertIn(col, stats.columns)
            
            # Check value ranges using .all() method
            self.assertTrue(
                ((stats['mean_acc'] >= 0) & (stats['mean_acc'] <= 1)).all(), 
                "Mean accuracy values must be between 0 and 1"
            )

    def test_theoretical_bounds(self):
        """Test calculation of theoretical performance bounds."""
        # Test WCST bound (3 tasks)
        wcst_bound = calculate_bounds(3, 6)
        self.assertAlmostEqual(wcst_bound, 0.75, places=2)
        
        # Test LNT bound (2 tasks)
        lnt_bound = calculate_bounds(2, 6)
        self.assertAlmostEqual(lnt_bound, 0.857, places=3)

    def test_plotting(self):
        """Test accuracy distribution plotting."""
        # Create mock result files
        self._create_mock_results()
        
        # Load results
        results = load_results(self.results_dir)
        
        # Skip test if seaborn is not available
        if sns is None:
            self.skipTest("Seaborn not installed, skipping plotting test")
        
        # Test plotting for each test
        for test_name, df in results.items():
            try:
                # Create plot
                plot_accuracy_distribution(df, test_name.upper(), self.output_dir)
                
                # Check if plot file was created
                plot_file = os.path.join(self.output_dir, f'{test_name.lower()}_distribution.png')
                self.assertTrue(os.path.exists(plot_file))
            except Exception as e:
                self.fail(f"Plotting failed for {test_name}: {e}")

    def test_component_task_analysis(self):
        """Test component task analysis."""
        # Create mock component task results
        os.makedirs(os.path.join(self.results_dir, 'component'), exist_ok=True)
        
        # Mock WCST component results
        wcst_tasks = ['shape', 'color', 'number']
        for task in wcst_tasks:
            mock_file = os.path.join(
                self.results_dir, 
                'component', 
                f'wcst_{task}_mock.json'
            )
            mock_results = [
                {"accuracy": 0.8, "model": "gpt-3.5-turbo"},
                {"accuracy": 0.9, "model": "gemini-1.5-pro"}
            ]
            with open(mock_file, 'w') as f:
                json.dump(mock_results, f)
        
        # Mock LNT component results
        lnt_tasks = ['letter', 'number']
        for task in lnt_tasks:
            mock_file = os.path.join(
                self.results_dir, 
                'component', 
                f'lnt_{task}_mock.json'
            )
            mock_results = [
                {"accuracy": 0.7, "model": "gpt-3.5-turbo"},
                {"accuracy": 0.85, "model": "gemini-1.5-pro"}
            ]
            with open(mock_file, 'w') as f:
                json.dump(mock_results, f)
        
        # Analyze component tasks
        # Skip test if seaborn is not available
        if sns is None:
            self.skipTest("Seaborn not installed, skipping component task analysis test")
        
        try:
            component_stats = analyze_component_tasks(
                os.path.join(self.results_dir, 'component'), 
                self.output_dir
            )
            
            # Verify stats structure
            self.assertIn('wcst_stats', component_stats)
            self.assertIn('lnt_stats', component_stats)
            
            # Check plot generation
            plot_file = os.path.join(self.output_dir, 'component_task_performance.png')
            self.assertTrue(os.path.exists(plot_file))
        except Exception as e:
            self.fail(f"Component task analysis failed: {e}")

    def test_full_analysis_pipeline(self):
        """Test the complete analysis pipeline."""
        # Create mock result files
        self._create_mock_results()
        
        try:
            # Run full analysis
            analyze_all(self.results_dir, self.output_dir)
            
            # Check generated files
            expected_files = [
                'wcst_stats.csv',
                'lnt_stats.csv',
                'wcst_distribution.png',
                'lnt_distribution.png'
            ]
            
            for file in expected_files:
                full_path = os.path.join(self.output_dir, file)
                self.assertTrue(os.path.exists(full_path), f"Missing file: {file}")
        except Exception as e:
            self.fail(f"Full analysis pipeline failed: {e}")

    def tearDown(self):
        """Clean up temporary directories."""
        # Remove temporary directories
        import shutil
        shutil.rmtree(self.temp_dir)

def run_analysis_tests():
    """Run all analysis tests."""
    unittest.main()

if __name__ == '__main__':
    run_analysis_tests()