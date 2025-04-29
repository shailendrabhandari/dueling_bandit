.. _methodology:

Methodology
===========

Problem Setting
---------------

The toolkit operates in a dueling bandits framework with :math:`k` items, where pairwise comparisons follow the Bradley-Terry-Luce (BTL) model. Each item :math:`i` has a score :math:`w_i`, and the probability of :math:`i` beating :math:`j` is:

.. math::

   P_{i,j} = \frac{w_i}{w_i + w_j}

Given a budget :math:`B`, the goal is to identify the item with the highest score using at most :math:`B` comparisons, typically under shoestring budgets (:math:`B = 2k, 3k, 4k`).

Algorithms
----------

The toolkit implements five algorithms:

- **Double Thompson Sampling (Double TS)** :cite:`NIPS2016_9de6d14f`: Uses two Thompson Sampling steps with Beta priors over pairwise preferences.
- **Random**: Selects pairs uniformly at random as a baseline.
- **PARWiS** :cite:`sheth2021parwis`: Combines an initialization phase (:math:`k-1` comparisons for spectral ranking) with disruptive pair selection to update rankings.
- **Contextual PARWiS**: Extends PARWiS with logistic regression to predict comparison outcomes using item features :cite:`xu2024linearcontextualbanditsinterference,datsai2022fastonlineinferencenonlinear`.
- **RL PARWiS**: Uses Q-learning for pair selection, with a state including ranking and comparison counts, actions as pair choices, and rewards combining regret reduction and winner recovery.

Datasets
--------

The toolkit supports three datasets:

- **Synthetic**: Generated via the BTL model with :math:`k=20` items and :math:`d=5` features. Scores are sampled from a normal distribution, and :math:`\Delta_{1,2} = 0.0152 \pm 0.0190`.
- **Jester** :cite:`goldberg2001eigentaste`: 4.1 million ratings for 100 jokes, with 20 selected (:math:`\Delta_{1,2} = 0.0946 \pm 0.0000`).
- **MovieLens** :cite:`harper2015movielens`: 20 million ratings for 27,000 movies, with 20 selected (:math:`\Delta_{1,2} = 0.0008 \pm 0.0000`).

Real-world datasets convert ratings to pairwise probabilities using a logistic function.

Evaluation Metrics
------------------

The toolkit uses the following metrics :cite:`sheth2021parwis`:

- **Recovery Fraction**: Fraction of runs where the true winner is recommended.
- **True Rank of Reported Winner**: True rank of the recommended item (lower is better).
- **Reported Rank of True Winner**: Rank of the true winner in the agent’s ranking (for PARWiS variants, lower is better).
- **Cumulative Regret**: Number of times a non-optimal item wins a duel.
- **:math:`\Delta_{1,2}`**: Separation between top two items, :math:`(P_{1,2} - 0.5)^2`, indicating problem difficulty.