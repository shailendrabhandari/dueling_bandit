from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from .environment import RankCentrality

@dataclass
class DoubleThompsonSamplingAgent:
    """Double Thompson Sampling agent for dueling bandits."""
    k: int
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.wins = np.zeros((self.k, self.k), dtype=int)
        self.losses = np.zeros((self.k, self.k), dtype=int)

    def select_pair(self) -> Tuple[int, int]:
        """Select a pair of items using Double Thompson Sampling."""
        scores = np.zeros(self.k)
        for i in range(self.k):
            wins_i = self.wins[i].sum()
            losses_i = self.losses[i].sum()
            scores[i] = self.rng.beta(wins_i + 1, losses_i + 1)
        a = int(np.argmax(scores))
        scores[a] = -np.inf
        b = int(np.argmax(scores))
        return a, b

    def update(self, a: int, b: int, winner: int):
        """Update win/loss counts based on duel outcome."""
        loser = b if winner == a else a
        self.wins[winner, loser] += 1
        self.losses[loser, winner] += 1

    def recommend(self) -> int:
        """Recommend the item with the most wins."""
        return int(np.argmax(self.wins.sum(axis=1)))

@dataclass
class RandomPairAgent:
    """Random pair selection agent for dueling bandits."""
    k: int
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def select_pair(self) -> Tuple[int, int]:
        """Select a random pair of items."""
        a, b = self.rng.choice(self.k, size=2, replace=False)
        return int(a), int(b)

    def update(self, a: int, b: int, winner: int):
        """No-op update (random agent doesn't track outcomes)."""
        pass

    def recommend(self) -> int:
        """Recommend a random item."""
        return int(self.rng.choice(self.k))

@dataclass
class PARWiSAgent:
    """PARWiS agent for winner determination under shoestring budgets."""
    k: int
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.wins = np.zeros((self.k, self.k), dtype=int)
        self.losses = np.zeros((self.k, self.k), dtype=int)
        self.rank = RankCentrality(self.k)
        self.pi = np.full(self.k, 1.0 / self.k)

    def _update_pi(self):
        """Update stationary distribution using spectral ranking."""
        self.pi = self.rank.stationary_distribution(self.wins, self.losses)

    def _disruption(self, i: int, j: int) -> float:
        """Calculate disruption score for a pair."""
        Nij = self.wins[i, j] + self.losses[i, j]
        Nji = self.wins[j, i] + self.losses[j, i]
        return (self.pi[i] * self.pi[j]) / (self.pi[i] + self.pi[j]) * (1.0 / (Nij + 1) + 1.0 / (Nji + 1))

    def phase1(self, env):
        """Initialize with k-1 comparisons."""
        order = self.rng.permutation(self.k)
        champ = int(order[0])
        for nxt in order[1:]:
            winner, loser = env.duel(champ, int(nxt))
            self.wins[winner, loser] += 1
            self.losses[loser, winner] += 1
            champ = winner
        self._update_pi()

    def select_pair(self) -> Tuple[int, int]:
        """Select the most disruptive pair."""
        disruptions = np.array([self._disruption(i, j) for i in range(self.k) for j in range(i + 1, self.k)])
        idx = np.argmax(disruptions)
        i = idx // (self.k - 1)
        j = idx % (self.k - 1) + (i + 1)
        return i, j

    def update(self, a: int, b: int, winner: int):
        """Update win/loss counts and ranking."""
        loser = b if winner == a else a
        self.wins[winner, loser] += 1
        self.losses[loser, winner] += 1
        self._update_pi()

    def recommend(self) -> int:
        """Recommend the top-ranked item."""
        return int(self.pi.argmax())

@dataclass
class ContextualPARWiSAgent:
    """Contextual PARWiS agent using logistic regression for feature-based predictions."""
    k: int
    d: int
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.wins = np.zeros((self.k, self.k), dtype=int)
        self.losses = np.zeros((self.k, self.k), dtype=int)
        self.rank = RankCentrality(self.k)
        self.pi = np.full(self.k, 1.0 / self.k)
        self.model = LogisticRegression(random_state=self.seed)
        self.X = []
        self.y = []

    def _update_pi(self):
        self.pi = self.rank.stationary_distribution(self.wins, self.losses)

    def _disruption(self, i: int, j: int, features: Optional[np.ndarray] = None) -> float:
        Nij = self.wins[i, j] + self.losses[i, j]
        Nji = self.wins[j, i] + self.losses[j, i]
        if features is not None and len(self.X) > 0:
            x = features[i] - features[j]
            try:
                prob = self.model.predict_proba([x])[0, 1]
                return (self.pi[i] * self.pi[j]) / (self.pi[i] + self.pi[j]) * prob
            except:
                pass
        return (self.pi[i] * self.pi[j]) / (self.pi[i] + self.pi[j]) * (1.0 / (Nij + 1) + 1.0 / (Nji + 1))

    def phase1(self, env):
        order = self.rng.permutation(self.k)
        champ = int(order[0])
        for nxt in order[1:]:
            winner, loser = env.duel(champ, int(nxt))
            if env.features is not None:
                x = env.features[champ] - env.features[nxt]
                self.X.append(x)
                self.y.append(1 if winner == champ else 0)
            self.wins[winner, loser] += 1
            self.losses[loser, winner] += 1
            champ = winner
        if len(self.X) > 0:
            self.model.fit(self.X, self.y)
        self._update_pi()

    def select_pair(self, features: Optional[np.ndarray] = None) -> Tuple[int, int]:
        disruptions = np.array([self._disruption(i, j, features) for i in range(self.k) for j in range(i + 1, self.k)])
        idx = np.argmax(disruptions)
        i = idx // (self.k - 1)
        j = idx % (self.k - 1) + (i + 1)
        return i, j

    def update(self, a: int, b: int, winner: int, features: Optional[np.ndarray] = None):
        loser = b if winner == a else a
        self.wins[winner, loser] += 1
        self.losses[loser, winner] += 1
        if features is not None:
            x = features[a] - features[b]
            self.X.append(x)
            self.y.append(1 if winner == a else 0)
            if len(self.X) >= self.k:
                self.model.fit(self.X, self.y)
        self._update_pi()

    def recommend(self) -> int:
        return int(self.pi.argmax())

@dataclass
class RLPARWiSAgent:
    """Reinforcement Learning PARWiS agent using Q-learning for pair selection."""
    k: int
    seed: Optional[int] = None
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    epsilon: float = 0.1

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.wins = np.zeros((self.k, self.k), dtype=int)
        self.losses = np.zeros((self.k, self.k), dtype=int)
        self.rank = RankCentrality(self.k)
        self.pi = np.full(self.k, 1.0 / self.k)
        self.q_table = np.zeros((self.k, self.k * (self.k - 1)))
        self.action_map = [(i, j) for i in range(self.k) for j in range(self.k) if i != j]

    def _update_pi(self):
        self.pi = self.rank.stationary_distribution(self.wins, self.losses)

    def _disruption(self, i: int, j: int) -> float:
        Nij = self.wins[i, j] + self.losses[i, j]
        Nji = self.wins[j, i] + self.losses[j, i]
        return (self.pi[i] * self.pi[j]) / (self.pi[i] + self.pi[j]) * (1.0 / (Nij + 1) + 1.0 / (Nji + 1))

    def _get_state(self) -> int:
        return int(self.pi.argmax())

    def _choose_action(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(len(self.action_map))
        return np.argmax(self.q_table[state])

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
        state = self._get_state()
        action_idx = self._choose_action(state)
        i, j = self.action_map[action_idx]
        return i, j

    def update(self, a: int, b: int, winner: int):
        loser = b if winner == a else a
        self.wins[winner, loser] += 1
        self.losses[loser, winner] += 1
        self._update_pi()
        state = self._get_state()
        action_idx = self.action_map.index((a, b))
        next_state = self._get_state()
        reward = 1.0 if winner == self.pi.argmax() else -0.1
        self.q_table[state, action_idx] += self.learning_rate * (
            reward + self.discount_factor * np.max(self.q_table[next_state]) - self.q_table[state, action_idx]
        )

    def recommend(self) -> int:
        return int(self.pi.argmax())