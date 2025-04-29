from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from .environment import RankCentrality

@dataclass
class RLPARWiSAgent:
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
        # Q-table: state (top item index) x action (pair to compare)
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
        reward = 1.0 if winner == self.pi.argmax() else -0.1  # Reward for selecting top item
        self.q_table[state, action_idx] += self.learning_rate * (
            reward + self.discount_factor * np.max(self.q_table[next_state]) - self.q_table[state, action_idx]
        )

    def recommend(self) -> int:
        return int(self.pi.argmax())