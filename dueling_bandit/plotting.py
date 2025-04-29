import os
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

METRIC_PROPERTIES = {
    'mean_regret': {
        'label': 'Cumulative Regret',
        'title': 'Cumulative Regret (Budget={budget}, Dataset={dataset})',
        'ylabel': 'Cumulative Regret',
        'std_key': 'std_regret',
        'ylim': None,
    },
    'mean_recovery': {
        'label': 'Recovery Fraction',
        'title': 'Recovery Fraction (Budget={budget}, Dataset={dataset})',
        'ylabel': 'Recovery Fraction',
        'std_key': 'std_recovery',
        'ylim': (0, 1),
    },
    'mean_true_rank': {
        'label': 'True Rank of Reported Winner',
        'title': 'True Rank of Reported Winner (Budget={budget}, Dataset={dataset})',
        'ylabel': 'True Rank of Reported Winner',
        'std_key': 'std_true_rank',
        'ylim': (0, 21),
    },
    'mean_reported_rank': {
        'label': 'Reported Rank of True Winner',
        'title': 'Reported Rank of True Winner (Budget={budget}, Dataset={dataset})',
        'ylabel': 'Reported Rank of True Winner',
        'std_key': 'std_reported_rank',
        'ylim': (0, 21),
        'filter': lambda mean: np.any(mean > 0),
    }
}

AGENT_STYLES = {
    'Double TS': {'color': 'blue', 'linestyle': '-'},
    'Random': {'color': 'orange', 'linestyle': '--'},
    'PARWiS': {'color': 'green', 'linestyle': '-'},
    'Contextual PARWiS': {'color': 'red', 'linestyle': '--'},
    'RL PARWiS': {'color': 'purple', 'linestyle': '-'}
}

def plot_metric(results: Dict[int, Dict[str, Dict[str, np.ndarray]]], 
                budget: int, 
                dataset: str,
                metric: str, 
                show_error_bars: bool = True, 
                save_path: Optional[str] = None):
    """
    Plot a single metric for all agents.

    Args:
        results: Dictionary of results per budget and agent.
        budget: Budget to plot.
        dataset: Dataset name.
        metric: Metric to plot (e.g., 'mean_regret').
        show_error_bars: Whether to show standard deviation error bars.
        save_path: Path to save the figure.
    """
    if metric not in METRIC_PROPERTIES:
        raise ValueError(f"Unknown metric: {metric}. Available: {list(METRIC_PROPERTIES.keys())}")

    props = METRIC_PROPERTIES[metric]
    plt.figure(figsize=(8, 6))

    for name, res in results[budget].items():
        if name == 'separation':
            continue
        mean = res[metric]
        std = res[props['std_key']]
        
        if 'filter' in props and not props['filter'](mean):
            continue

        x = range(len(mean))
        style = AGENT_STYLES.get(name, {'color': 'black', 'linestyle': '-'})
        plt.plot(x, mean, label=name, color=style['color'], linestyle=style['linestyle'])
        if show_error_bars:
            plt.fill_between(x, mean - std, mean + std, color=style['color'], alpha=0.2)

    plt.xlabel("Duels")
    plt.ylabel(props['ylabel'])
    plt.title(props['title'].format(budget=budget, dataset=dataset))
    if props['ylim']:
        plt.ylim(props['ylim'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_metrics(results: Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]], 
                     datasets: List[str], 
                     budgets: List[int], 
                     metrics: Optional[List[str]] = None, 
                     save_dir: Optional[str] = None):
    """
    Plot all metrics for all datasets and budgets.

    Args:
        results: Dictionary of results per dataset, budget, and agent.
        datasets: List of dataset names.
        budgets: List of budgets.
        metrics: List of metrics to plot (default: all).
        save_dir: Directory to save plots.
    """
    metrics = metrics or list(METRIC_PROPERTIES.keys())
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for dataset in datasets:
        for B in budgets:
            for metric in metrics:
                save_path = f"{save_dir}/{dataset}_{metric}_budget_{B}.png" if save_dir else None
                plot_metric(results[dataset], B, dataset, metric, save_path=save_path)