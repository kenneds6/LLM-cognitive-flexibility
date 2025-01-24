"""Analysis for component task performance."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_component_tasks(component_results_dir: str, 
                          output_dir: str = 'analysis_output'):
    """Analyze and visualize component task performance."""
    results_path = Path(component_results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load WCST component results
    wcst_tasks = ['shape', 'color', 'number']
    wcst_results = []
    
    for task in wcst_tasks:
        files = list(results_path.glob(f"wcst_{task}_*.json"))
        for file in files:
            model = file.stem.split('_')[2]
            with open(file) as f:
                data = pd.read_json(f)
                data['model'] = model
                data['task'] = task
                wcst_results.append(data)
    
    wcst_df = pd.concat(wcst_results)
    
    # Load LNT component results
    lnt_tasks = ['letter', 'number']
    lnt_results = []
    
    for task in lnt_tasks:
        files = list(results_path.glob(f"lnt_{task}_*.json"))
        for file in files:
            model = file.stem.split('_')[2]
            with open(file) as f:
                data = pd.read_json(f)
                data['model'] = model
                data['task'] = task
                lnt_results.append(data)
    
    lnt_df = pd.concat(lnt_results)
    
    # Generate component task plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=wcst_df, x='task', y='accuracy', hue='model')
    plt.title('WCST Component Task Performance')
    plt.xlabel('Task')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=lnt_df, x='task', y='accuracy', hue='model')
    plt.title('LNT Component Task Performance')
    plt.xlabel('Task')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path / 'component_task_performance.png')
    plt.close()
    
    # Generate summary statistics
    wcst_stats = wcst_df.groupby(['model', 'task'])['accuracy'].agg(['mean', 'std'])
    lnt_stats = lnt_df.groupby(['model', 'task'])['accuracy'].agg(['mean', 'std'])
    
    return {
        'wcst_stats': wcst_stats,
        'lnt_stats': lnt_stats
    }

if __name__ == "__main__":
    stats = analyze_component_tasks('results')