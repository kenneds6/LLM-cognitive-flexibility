"""Analysis utilities for cognitive flexibility experiments."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

def load_results(results_dir: str) -> Dict[str, pd.DataFrame]:
    """Load results from JSON files into DataFrames."""
    results_path = Path(results_dir)
    dfs = {}
    
    for test in ['wcst', 'lnt']:
        test_files = list(results_path.glob(f"{test}_*.json"))
        all_results = []
        
        for file in test_files:
            model = file.stem.split('_')[1]
            with open(file) as f:
                data = json.load(f)
                for result in data:
                    result['model'] = model
                all_results.extend(data)
        
        dfs[test] = pd.DataFrame(all_results)
    
    return dfs

def calculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean accuracy and standard deviation by model."""
    stats = df.groupby('model').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'score': ['mean', 'std'],
        'trials': ['mean']
    }).round(4)
    
    stats.columns = ['mean_acc', 'std_acc', 'min_acc', 'max_acc', 
                    'mean_score', 'std_score', 'avg_trials']
    return stats

def calculate_bounds(num_tasks: int, required_successes: int) -> float:
    """Calculate worst-case performance bound."""
    return required_successes / (required_successes + (num_tasks - 1))

def plot_accuracy_distribution(df: pd.DataFrame, test_name: str, 
                             output_dir: str = 'figures'):
    """Create boxplot of accuracy distributions."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='model', y='accuracy')
    plt.title(f'{test_name} Accuracy Distribution by Model')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    # Add theoretical bound
    num_tasks = 3 if test_name == 'WCST' else 2
    bound = calculate_bounds(num_tasks, 6)
    plt.axhline(y=bound, color='r', linestyle='--', 
                label=f'Theoretical Bound ({bound:.2f})')
    plt.legend()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / f'{test_name.lower()}_distribution.png', 
                bbox_inches='tight')
    plt.close()

def analyze_all(results_dir: str = 'results', 
                output_dir: str = 'analysis_output'):
    """Run complete analysis pipeline."""
    # Load results
    results = load_results(results_dir)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Analyze each test
    for test_name, df in results.items():
        # Calculate statistics
        stats = calculate_statistics(df)
        stats.to_csv(output_path / f'{test_name}_stats.csv')
        
        # Generate plots
        plot_accuracy_distribution(df, test_name.upper(), output_dir)
        
        # Print summary
        print(f"\n{test_name.upper()} Results:")
        print(stats)
        
        # Calculate theoretical bound
        num_tasks = 3 if test_name == 'wcst' else 2
        bound = calculate_bounds(num_tasks, 6)
        print(f"Theoretical performance bound: {bound:.4f}")

if __name__ == "__main__":
    analyze_all()