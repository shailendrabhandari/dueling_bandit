# Dueling Bandit Toolkit

A Python package for preference-based online learning with dueling bandits. This toolkit implements algorithms like Double Thompson Sampling, PARWiS, and Contextual PARWiS, with support for real-world datasets (Jester, MovieLens) and evaluation metrics.
Features

Algorithms: Double Thompson Sampling, PARWiS, Contextual PARWiS, and Random Pair baseline.
Environment: Bradley-Terry model with optional contextual features.
Datasets: Synthetic, Jester, and MovieLens support.
Metrics: Cumulative regret, recovery fraction, true/reported ranks, and separation (Δ_1,2).
Visualization: Plotting functions for experiment results.

Installation
pip install dueling-bandit

Usage
from dueling_bandit.environment import DuelingBanditEnv
from dueling_bandit.agents import DoubleThompsonSamplingAgentpython -m build
from dueling_bandit.experiments import run_simulation

# Create environment
env = DuelingBanditEnv.random_bt(k=20, d=5, seed=42)

# Initialize agent
agent = DoubleThompsonSamplingAgent(k=20, seed=42)

# Run simulation
results = run_simulation(env, agent, horizon=500)

# Plot results (requires matplotlib)
from dueling_bandit.plotting import plot_metric
plot_metric({'500': {'Double TS': results}}, budget=500, dataset='synthetic', metric='mean_regret')

Requirements

Python >= 3.8
numpy, matplotlib, scipy, pandas

Install dependencies:
pip install -r requirements.txt

Development

Clone the repository:git clone https://github.com/shailendrabhandari/dueling_bandit.git
cd dueling-bandit


Install in editable mode:pip install -e .


Run tests:pytest tests/



Documentation
Full documentation is available at ReadTheDocs.
License
MIT License. See LICENSE for details.
Contributing
Contributions are welcome! Please open an issue or pull request on GitHub.
