from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.special import expit
from .environment import RankCentrality

@dataclass
class DoubleThompsonSamplingAgent:
    k: int
    alpha: float = 1.0
    beta: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self):
        self.wins = np.zeros((self.k, self.k), dtype=int)
        self.losses = np.zeros((self.k, self.k), dtype=int)
        self.rng = np.random.default_rng(self.seed)

    def _sample_theta(self) -> np.ndarray:
        theta = np.empty((self.k, self.k))
        for i in range(self.k):
            for j in range(self.k):
                if i == j:
                    theta[i, j] = 0.5
                else:
                    a = self.wins[i, j] + self.alpha
                    b = self.losses[i, j] + self.beta
                    theta[i, j] = self.rng.beta(a, b)
        return theta

    def _tournament_winner(self, theta: np.ndarray) -> int:
        scores = (theta > 0.5).sum(axis=1)
        best_score = scores.max()
        candidates = np.where(scores == best_score)[0]
        if len(candidates) == 1:
            return int(candidates[0])
        avg_pref = theta.mean(axis=1)
        return int(candidates[np.argmax(avg_pref[candidates])])

    def select_pair(self) -> Tuple[int, int]:
        theta1 = self._sample_theta()
        a = self._tournament_winner(theta1)
        theta2 = self._sample_theta()
        b = self._tournament_winner(theta2)
        while b == a:
            b = int(self.rng.integers(0, self.k))
        return a, b

    def update(self, a: int, b: int, winner: int):
        if winner == a:
            self.wins[a, b] += 1
            self.losses[b, a] += 1
        else:
            self.wins[b, a] += 1
            self.losses[a, b] += 1

    def recommend(self) -> int:
        theta = self._sample_theta()
        return self._tournament_winner(theta)

@dataclass
class RandomPairAgent:
    k: int
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def select_pair(self) -> Tuple[int, int]:
        a, b = self.rng.choice(self.k, size=2, replace=False)
        return int(a), int(b)

    def update(self, *_) -> None:
        pass

    def recommend(self) -> int:
        return int(self.rng.integers(0, self.k))

@dataclass
class PARWiSAgent:
    k: int
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.wins = np.zeros((self.k, self.k), dtype=int)
        self.losses = np.zeros((self.k, self.k), dtype=int)
        self.rank = RankCentrality(self.k)
        self.pi = np.full(self.k, 1.0 / self.k)

    def _update_pi(self):
        self.pi = self.rank.stationary_distribution(self.wins, self.losses)

    def _disruption(self, i: int, j: int) -> float:
        Nij = self.wins[i, j] + self.losses[i, j]
        Nji = self.wins[j, i] + self.losses[j, i]
        return (self.pi[i] * self.pi[j]) / (self.pi[i] + self.pi[j]) * (1.0 / (Nij + 1) + 1.0 / (Nji + 1))

    def phase1(self, env):
        order = self.rng.permutation(self.k)
        champ = int(order[0])
        for nxt in order[1:]:
            winner, loser = env.duel(champ, int(nxt))
            self.wins[winner, loser] += 1
            self.losses[loser, winner] += 1
            champ = winner
        self._update_pi()

    def select_pair(self) -> Tuple[int, int]:
        top = int(self.pi.argmax())
        j = int(np.argmax([self._disruption(top, x) if x != top else -1.0 for x in range(self.k)]))
        return top, j

    def update(self, a: int, b: int, winner: int):
        loser = b if winner == a else a
        self.wins[winner, loser] += 1
        self.losses[loser, winner] += 1
        self._update_pi()

    def recommend(self) -> int:
        return int(self.pi.argmax())

@dataclass
class ContextualPARWiSAgent:
    k: int
    d: int
    lr: float = 0.01
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.wins = np.zeros((self.k, self.k), dtype=int)
        self.losses = np.zeros((self.k, self.k), dtype=int)
        self.rank = RankCentrality(self.k)
        self.pi = np.full(self.k, 1.0 / self.k)
        self.weights = np.zeros(self.d)
        self.features = None

    def set_features(self, features: np.ndarray):
        assert features.shape == (self.k, self.d), "Feature shape mismatch"
        self.features = features

    def _predict_prob(self, i: int, j: int) -> float:
        if self.features is None:
            return self.pi[i] / (self.pi[i] + self.pi[j])
        diff = self.features[i] - self.features[j]
        return expit(np.dot(self.weights, diff))

    def _update_weights(self, i: int, j: int, winner: int):
        pred_prob = self._predict_prob(i, j)
        y = 1.0 if winner == i else 0.0
        diff = self.features[i] - self.features[j]
        grad = (pred_prob - y) * diff
        self.weights -= self.lr * grad

    def _update_pi(self):
        self.pi = self.rank.stationary_distribution(self.wins, self.losses)

    def _disruption(self, i: int, j: int) -> float:
        Nij = self.wins[i, j] + self.losses[i, j]
        Nji = self.wins[j, i] + self.losses[j, i]
        contextual_prob = self._predict_prob(i, j)
        return (self.pi[i] * self.pi[j]) / (self.pi[i] + self.pi[j]) * (1.0 / (Nij + 1) + 1.0 / (Nji + 1)) * contextual_prob

    def phase1(self, env):
        self.set_features(env.features)
        order = self.rng.permutation(self.k)
        champ = int(order[0])
        for nxt in order[1:]:
            winner, loser = env.duel(champ, int(nxt))
            self.wins[winner, loser] += 1
            self.losses[loser, winner] += 1
            self._update_weights(champ, nxt, winner)
            champ = winner
        self._update_pi()

    def select_pair(self) -> Tuple[int, int]:
        top = int(self.pi.argmax())
        j = int(np.argmax([self._disruption(top, x) if x != top else -1.0 for x in range(self.k)]))
        return top, j

    def update(self, a: int, b: int, winner: int):
        loser = b if winner == a else a
        self.wins[winner, loser] += 1
        self.losses[loser, winner] += 1
        self._update_weights(a, b, winner)
        self._update_pi()

    def recommend(self) -> int:
        return int(self.pi.argmax())