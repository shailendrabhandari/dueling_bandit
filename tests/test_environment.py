import pytest
import numpy as np
from dueling_bandit.environment import DuelingBanditEnv

def test_dueling_bandit_env():
    k = 5
    env = DuelingBanditEnv.random_bt(k, seed=42)
    assert env.k == k
    assert env.P.shape == (k, k)
    assert np.allclose(np.diag(env.P), 0.5)
    
    winner, loser = env.duel(0, 1)
    assert winner in [0, 1]
    assert loser in [0, 1]
    assert winner != loser
    
    best = env.best_arm()
    assert 0 <= best < k
    
    true_order = env.true_order()
    assert len(true_order) == k
    assert sorted(true_order) == list(range(k))
    
    sep = env.separation()
    assert sep >= 0