.. _experiments:

Experiments
===========

Setup
-----

The toolkit evaluates algorithms with :math:`k=20` items, budgets :math:`B \in \{40, 60, 80\}`, and 30 runs per dataset. Synthetic data includes random features, while Jester and MovieLens lack features, causing Contextual PARWiS to fall back to non-contextual behavior. RL PARWiS is trained for 5000 episodes.

Results
-------

The following tables summarize recovery fraction, true rank of reported winner, and cumulative regret across datasets and budgets, as reported in :cite:`sheth2021parwis`.

.. list-table:: Recovery Fraction
   :widths: 20 15 15 15 15 15 15 15 15 15
   :header-rows: 1
   :name: recovery_comparison

   * - Agent
     - Synthetic (B=40)
     - Synthetic (B=60)
     - Synthetic (B=80)
     - Jester (B=40)
     - Jester (B=60)
     - Jester (B=80)
     - MovieLens (B=40)
     - MovieLens (B=60)
     - MovieLens (B=80)
   * - Double TS
     - 0.200
     - 0.067
     - 0.267
     - 0.167
     - 0.233
     - 0.467
     - 0.133
     - 0.067
     - 0.067
   * - Random
     - 0.033
     - 0.067
     - 0.000
     - 0.033
     - 0.000
     - 0.067
     - 0.033
     - 0.000
     - 0.067
   * - PARWiS
     - **0.467**
     - **0.467**
     - **0.467**
     - **0.467**
     - **0.467**
     - **0.467**
     - **0.167**
     - **0.167**
     - **0.167**
   * - Contextual PARWiS
     - 0.367
     - 0.367
     - 0.367
     - 0.433
     - 0.433
     - 0.433
     - **0.167**
     - **0.167**
     - **0.167**
   * - RL PARWiS
     - 0.367
     - 0.367
     - 0.367
     - **0.467**
     - **0.467**
     - **0.467**
     - 0.100
     - 0.100
     - 0.100

.. list-table:: True Rank of Reported Winner
   :widths: 20 15 15 15 15 15 15 15 15 15
   :header-rows: 1
   :name: truerank_comparison

   * - Agent
     - Synthetic (B=40)
     - Synthetic (B=60)
     - Synthetic (B=80)
     - Jester (B=40)
     - Jester (B=60)
     - Jester (B=80)
     - MovieLens (B=40)
     - MovieLens (B=60)
     - MovieLens (B=80)
   * - Double TS
     - 8.233
     - 6.933
     - 4.767
     - 6.700
     - 4.700
     - 3.133
     - 9.233
     - 10.300
     - 11.500
   * - Random
     - 10.767
     - 10.367
     - 10.733
     - 10.733
     - 9.367
     - 10.733
     - 9.233
     - 11.033
     - 10.767
   * - PARWiS
     - **3.233**
     - **3.233**
     - **3.233**
     - **2.067**
     - **2.067**
     - **2.067**
     - **6.633**
     - **6.633**
     - **6.633**
   * - Contextual PARWiS
     - 3.900
     - 4.067
     - 4.067
     - 2.233
     - 2.233
     - 2.233
     - **6.633**
     - **6.633**
     - **6.633**
   * - RL PARWiS
     - 3.533
     - 3.533
     - 3.533
     - **2.067**
     - **2.067**
     - **2.067**
     - 6.667
     - 6.667
     - 6.667

.. list-table:: Cumulative Regret
   :widths: 20 15 15 15 15 15 15 15 15 15
   :header-rows: 1
   :name: regret_comparison

   * - Agent
     - Synthetic (B=40)
     - Synthetic (B=60)
     - Synthetic (B=80)
     - Jester (B=40)
     - Jester (B=60)
     - Jester (B=80)
     - MovieLens (B=40)
     - MovieLens (B=60)
     - MovieLens (B=80)
   * - Double TS
     - 35.300
     - 52.933
     - 67.267
     - 34.067
     - 51.167
     - 67.667
     - 36.733
     - 55.767
     - 74.800
   * - Random
     - 36.633
     - 54.833
     - 73.200
     - 36.167
     - 54.233
     - 72.600
     - 37.733
     - 56.767
     - 75.800
   * - PARWiS
     - **11.733**
     - **22.000**
     - **33.133**
     - **9.567**
     - **17.600**
     - **25.633**
     - **18.067**
     - **35.100**
     - **52.333**
   * - Contextual PARWiS
     - 13.067
     - 24.333
     - 35.467
     - 10.167
     - 18.533
     - 27.200
     - 18.100
     - 35.133
     - 52.367
   * - RL PARWiS
     - 14.367
     - 26.300
     - 42.300
     - 11.100
     - 21.667
     - 32.400
     - 19.567
     - 38.633
     - 56.967

**Discussion**:

- **Synthetic and Jester**: PARWiS and RL PARWiS achieve high recovery fractions (0.467) and low true ranks (2.067–3.233), excelling on datasets with moderate to large :math:`\Delta_{1,2}` (0.0152 for Synthetic, 0.0946 for Jester). Cumulative regret is significantly lower for PARWiS (e.g., 11.733 at :math:`B=40` on Synthetic).
- **MovieLens**: The small :math:`\Delta_{1,2}` (0.0008) challenges all algorithms, with recovery fractions dropping to 0.100–0.167. PARWiS maintains the lowest regret (52.333 at :math:`B=80`).
- **Contextual PARWiS**: Performs similarly to PARWiS on real-world datasets due to missing features, with slight underperformance on Synthetic data, suggesting feature optimization needs.
- **RL PARWiS**: Matches PARWiS on Jester but struggles on MovieLens, indicating potential for improved training or state representation.

Statistical t-tests (see :ref:`Appendix Tables <appendix_tables>`) confirm PARWiS’s significant improvements over Double TS on Synthetic and Jester (:math:`p < 0.05`), but not on MovieLens due to the dataset’s difficulty.