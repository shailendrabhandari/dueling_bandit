# Dueling Bandit Toolkit

[![Documentation Status](https://readthedocs.org/projects/dueling-bandit/badge/?version=latest)](https://dueling-bandit.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen)](https://github.com/shailendrabhandari/dueling_bandit/blob/main/LICENSE)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dueling-bandit)](https://pypi.org/project/dueling-bandit/)
[![PyPI](https://img.shields.io/pypi/v/dueling-bandit)](https://pypi.org/project/dueling-bandit/)
[![Downloads](https://pepy.tech/badge/dueling-bandit)](https://pepy.tech/project/dueling-bandit)
![GitHub watchers](https://img.shields.io/github/watchers/shailendrabhandari/dueling_bandit?style=social)
![GitHub stars](https://img.shields.io/github/stars/shailendrabhandari/dueling_bandit?style=social)

The **Dueling Bandit Toolkit** is a Python package designed for preference-based online learning using dueling bandit algorithms. It provides robust implementations of state-of-the-art algorithms, support for real-world datasets, and comprehensive evaluation metrics, making it an ideal tool for researchers and practitioners in machine learning and decision-making systems.

## Features

- **Algorithms**: Includes Double Thompson Sampling, PARWiS, Contextual PARWiS, and a Random Pair baseline.
- **Environment**: Supports the Bradley-Terry model with optional contextual features for flexible experimentation.
- **Datasets**: Compatible with synthetic data and real-world datasets like Jester and MovieLens.
- **Metrics**: Evaluates performance with cumulative regret, recovery fraction, true/reported ranks, and separation (Δ₁,₂).
- **Visualization**: Offers plotting functions to visualize experiment results using Matplotlib.

## Installation

Install the package via pip from [PyPI](https://pypi.org/project/dueling-bandit/):

```bash
pip install dueling-bandit
```

Ensure you have Python 3.8 or higher.

## Quick Start

Here's a simple example to get started with the toolkit:

```python
from dueling_bandit.environment import DuelingBanditEnv
from dueling_bandit.agents import DoubleThompsonSamplingAgent
from dueling_bandit.experiments import run_simulation
from dueling_bandit.plotting import plot_metric

# Create a synthetic Bradley-Terry environment
env = DuelingBanditEnv.random_bt(k=20, d=5, seed=42)

# Initialize the Double Thompson Sampling agent
agent = DoubleThompsonSamplingAgent(k=20, seed=42)

# Run a simulation for 500 duels
results = run_simulation(env, agent, horizon=500)

# Visualize the cumulative regret
plot_metric({'500': {'Double TS': results}}, budget=500, dataset='synthetic', metric='mean_regret')
```

This code sets up a synthetic environment, runs a simulation with the Double Thompson Sampling algorithm, and plots the cumulative regret.

## Requirements

- Python >= 3.8
- Dependencies: `numpy`, `matplotlib`, `scipy`, `pandas`

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Development

To contribute or experiment with the toolkit:

1. Clone the repository:
   ```bash
   git clone https://github.com/shailendrabhandari/dueling_bandit.git
   cd dueling_bandit
   ```
2. Install in editable mode:
   ```bash
   pip install -e .
   ```
3. Run tests to ensure everything works:
   ```bash
   pytest tests/
   ```

## Documentation

Comprehensive documentation is available at [ReadTheDocs](https://dueling-bandit.readthedocs.io/en/latest/). It includes detailed API references, tutorials, and examples to help you get the most out of the toolkit.

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Whether it's bug fixes, new features, or documentation improvements, please:
1. Open an issue to discuss your idea.
2. Submit a pull request with your changes.

See the [Contributing Guidelines](https://github.com/shailendrabhandari/dueling_bandit/blob/main/CONTRIBUTING.md) for more details.

## Contact

For questions or support, please open an issue on [GitHub](https://github.com/shailendrabhandari/dueling_bandit/issues) or contact Shailendra at [shailendra.bhandari@oslomet.no](mailto:shailendra.bhandari@oslomet.no).

---

*Happy dueling!*