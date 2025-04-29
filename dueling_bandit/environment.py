from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigs

class RankCentrality:
    """Rank Centrality for spectral ranking from pairwise comparisons."""
    def __init__(self, k: int):
        self.k = k

    def stationary_distribution(self, wins: np.ndarray, losses: np.ndarray) -> np.ndarray:
        """
        Compute stationary distribution using Rank Centrality.

        Args:
            wins (np.ndarray): Win counts (k x k).
            losses (np.ndarray): Loss counts (k x k).

        Returns:
            np.ndarray: Stationary distribution (pi).
        """
        total = wins + losses
        total = np.where(total == 0, 1, total)
        P = wins / total
        P = np.where(wins + losses == 0, 1.0 / self.k, P)
        np.fill_diagonal(P, 0)
        degrees = P.sum(axis=1)
        degrees = np.where(degrees == 0, 1, degrees)
        D_inv = diags(1.0 / degrees)
        M = eye(self.k) - D_inv @ P
        _, vecs = eigs(M.T, k=1, which='SM')
        pi = np.abs(vecs[:, 0].real)
        pi /= pi.sum()
        return pi

@dataclass
class DuelingBanditEnv:
    """Dueling Bandit environment with Bradley-Terry model."""
    P: np.ndarray
    features: Optional[np.ndarray] = None
    seed: Optional[int] = None

    def __post_init__(self):
        self.k = self.P.shape[0]
        self.rng = np.random.default_rng(self.seed)
        self.rank = RankCentrality(self.k)

    @classmethod
    def random_bt(cls, k: int, d: int = 0, seed: Optional[int] = None) -> "DuelingBanditEnv":
        """Generate a random Bradley-Terry environment."""
        rng = np.random.default_rng(seed)
        utilities = rng.normal(size=k)
        P = 1.0 / (1.0 + np.exp(utilities[:, None] - utilities[None, :]))
        np.fill_diagonal(P, 0.5)
        features = rng.normal(size=(k, d)) if d > 0 else None
        return cls(P, features, seed)

    def duel(self, a: int, b: int) -> Tuple[int, int]:
        """Perform a duel between two items."""
        if self.rng.random() < self.P[a, b]:
            return a, b
        return b, a

    def best_arm(self) -> int:
        """Return the index of the item with the highest BTL score."""
        utilities = -np.log(1.0 / self.P - 1)
        return int(np.argmax(np.nanmean(utilities, axis=1)))

    def true_rank(self) -> np.ndarray:
        """Return the true ranking of items."""
        utilities = -np.log(1.0 / self.P - 1)
        scores = np.nanmean(utilities, axis=1)
        return np.argsort(-scores)

    def delta12(self) -> float:
        """Compute the separation metric Delta_1,2."""
        top2 = np.argsort(-np.nanmean(-np.log(1.0 / self.P - 1), axis=1))[:2]
        return (self.P[top2[0], top2[1]] - 0.5) ** 2