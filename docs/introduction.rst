.. _introduction:

Introduction
============

Overview
--------

Determining a winner among a set of items using active pairwise comparisons under a limited budget is a challenging problem in preference-based learning. The **Dueling Bandit Toolkit** implements the PARWiS algorithm, proposed by Sheth and Rajkumar :cite:`sheth2021parwis`, which leverages spectral ranking and disruptive pair selection to identify the best item under shoestring budgets. The toolkit extends PARWiS with two novel variants: Contextual PARWiS, incorporating item features, and RL PARWiS, using reinforcement learning for pair selection. These are compared against baselines like Double Thompson Sampling :cite:`NIPS2016_9de6d14f` and a random selection strategy.

The toolkit evaluates performance on synthetic and real-world datasets (Jester :cite:`goldberg2001eigentaste` and MovieLens :cite:`harper2015movielens`) using budgets of 40, 60, and 80 comparisons for 20 items. Metrics include recovery fraction, true rank of reported winner, reported rank of true winner, cumulative regret, and the separation metric :math:`\Delta_{1,2}`. Results show that PARWiS and RL PARWiS outperform baselines, particularly on the Jester dataset with a higher :math:`\Delta_{1,2}`, while Contextual PARWiS shows comparable performance, indicating potential for further feature optimization.

Background
----------

Preference-based learning through pairwise comparisons is widely used in recommender systems, social choice, and information retrieval :cite:`10.1007/978-3-319-91908-9_4,10.1007/s10994-019-05867-2`. In scenarios with limited comparison budgets (shoestring budgets), efficient algorithms are critical. PARWiS addresses this by using spectral ranking :cite:`negahban2016rank` and active pair selection, building on the Bradley-Terry-Luce (BTL) model where the probability of item :math:`i` beating item :math:`j` is:

.. math::

   P_{i,j} = \frac{w_i}{w_i + w_j}

This toolkit provides a practical implementation of PARWiS and its extensions, enabling researchers to explore winner determination under constrained budgets.