import numpy as np
from typing import Optional, Tuple, List
from scipy.special import expit

class DuelingBanditEnv:
    """Fixed preference matrix environment with optional arm features."""
    def __init__(self, preference_matrix: np.ndarray, features: Optional[np.ndarray] = None, seed: Optional[int] = None):
        assert preference_matrix.shape[0] == preference_matrix.shape[1], "Preference matrix must be square"
        self.P = preference_matrix
        self.k = preference_matrix.shape[0]
        self.features = features
        self.rng = np.random.default_rng(seed)

    def duel(self, a: int, b: int) -> Tuple[int, int]:
        """Play a duel between arms *a* and *b* and return (winner, loser)."""
        assert a != b, "Arms must be distinct"
        if self.rng.random() < self.P[a, b]:
            return a, b
        return b, a

    @staticmethod
    def random_bt(k: int, d: int = 0, seed: Optional[int] = None) -> "DuelingBanditEnv":
        """Bradley-Terry generator with optional features."""
        rng = np.random.default_rng(seed)
        utilities = rng.normal(size=k)
        P = np.empty((k, k))
        for i in range(k):
            for j in range(k):
                P[i, j] = 0.5 if i == j else 1.0 / (1.0 + np.exp(utilities[j] - utilities[i]))
        features = rng.normal(size=(k, d)) if d > 0 else None
        return DuelingBanditEnv(P, features, seed)

    @staticmethod
    def from_ratings(ratings: np.ndarray, k: int, seed: Optional[int] = None) -> "DuelingBanditEnv":
        """Create a DuelingBanditEnv from a ratings matrix (items x users)."""
        avg_ratings = np.nanmean(ratings, axis=1)
        P = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if i == j:
                    P[i, j] = 0.5
                else:
                    diff = avg_ratings[i] - avg_ratings[j]
                    P[i, j] = expit(diff)
        features = None
        return DuelingBanditEnv(P, features, seed)

    def condorcet_winner(self) -> Optional[int]:
        wins = (self.P > 0.5).sum(axis=1)
        if wins.max() == self.k - 1:
            return int(wins.argmax())
        return None

    def best_arm(self) -> int:
        cw = self.condorcet_winner()
        return cw if cw is not None else int(self.P.mean(axis=1).argmax())

    def true_order(self) -> List[int]:
        return list(np.argsort(-self.P.mean(axis=1)))

    def separation(self) -> float:
        """Compute Δ_1,2 = (P_1,2 - 0.5)^2 for the top two arms."""
        true_order = self.true_order()
        top, second = true_order[0], true_order[1]
        return (self.P[top, second] - 0.5) ** 2

class RankCentrality:
    def __init__(self, k: int):
        self.k = k

    def stationary_distribution(self, wins: np.ndarray, losses: np.ndarray, tol: float = 1e-10, iters: int = 1000) -> np.ndarray:
        """Return stationary distribution of augmented Markov chain."""
        N = wins + losses
        d_max = max(N.max(initial=1), 1)
        Q = np.zeros((self.k + 1, self.k + 1))
        for i in range(self.k):
            for j in range(self.k):
                if i != j:
                    Q[i + 1, j + 1] = losses[i, j] / d_max
        for i in range(self.k):
            Q[0, i + 1] = Q[i + 1, 0] = 1.0 / d_max
        Q[np.diag_indices_from(Q)] = 0.0
        row_sums = Q.sum(axis=1, keepdims=True)
        Q = np.divide(Q, row_sums, out=np.zeros_like(Q), where=row_sums != 0)
        Q[np.diag_indices_from(Q)] = 1.0 - Q.sum(axis=1)
        p = np.full(self.k + 1, 1.0 / (self.k + 1))
        for _ in range(iters):
            p_new = p @ Q
            if np.linalg.norm(p_new - p, 1) < tol:
                break
            p = p_new
        return p[1:] / p[1:].sum()