from typing import Dict, List, Optional
import numpy as np
from .environment import DuelingBanditEnv
from .agents import DoubleThompsonSamplingAgent, RandomPairAgent, PARWiSAgent, ContextualPARWiSAgent, RLPARWiSAgent
from .datasets import load_jester_data, load_movielens_data, ratings_to_preference_matrix

def run_simulation(env: DuelingBanditEnv, agent, horizon: int) -> Dict[str, np.ndarray]:
    """
    Run a single simulation with the given environment and agent.

    Args:
        env (DuelingBanditEnv): The dueling bandit environment.
        agent: The agent to evaluate.
        horizon (int): Number of duels to perform.

    Returns:
        Dict[str, np.ndarray]: Metrics (mean_regret, std_regret, mean_recovery, etc.).
    """
    k = env.k
    true_rank = env.true_rank()
    true_winner = true_rank[0]
    regret = np.zeros(horizon)
    recovery = np.zeros(horizon)
    true_rank_reported = np.zeros(horizon)
    reported_rank_true = np.zeros(horizon)

    if hasattr(agent, 'phase1'):
        agent.phase1(env)
        init_comparisons = k - 1
        for t in range(min(init_comparisons, horizon)):
            regret[t] = 0  # Initial phase regret
            recovery[t] = 1 if agent.recommend() == true_winner else 0
            true_rank_reported[t] = np.where(true_rank == agent.recommend())[0][0] + 1
            if hasattr(agent, 'pi'):
                reported_rank_true[t] = np.argsort(-agent.pi).tolist().index(true_winner) + 1
    else:
        init_comparisons = 0

    for t in range(init_comparisons, horizon):
        a, b = agent.select_pair() if not hasattr(agent, 'features') else agent.select_pair(env.features)
        winner, loser = env.duel(a, b)
        agent.update(a, b, winner, env.features)
        regret[t] = 1 if winner != true_winner else 0
        recovery[t] = 1 if agent.recommend() == true_winner else 0
        true_rank_reported[t] = np.where(true_rank == agent.recommend())[0][0] + 1
        if hasattr(agent, 'pi'):
            reported_rank_true[t] = np.argsort(-agent.pi).tolist().index(true_winner) + 1

    return {
        'mean_regret': np.cumsum(regret),
        'std_regret': np.std(regret) * np.ones(horizon),
        'mean_recovery': recovery,
        'std_recovery': np.std(recovery) * np.ones(horizon),
        'mean_true_rank': true_rank_reported,
        'std_true_rank': np.std(true_rank_reported) * np.ones(horizon),
        'mean_reported_rank': reported_rank_true,
        'std_reported_rank': np.std(reported_rank_true) * np.ones(horizon)
    }

def evaluate(k: int, horizon: int, dataset: str = 'synthetic', n_runs: int = 30, seed: Optional[int] = None) -> Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]]:
    """
    Evaluate all agents on the specified dataset.

    Args:
        k (int): Number of items.
        horizon (int): Budget for comparisons.
        dataset (str): Dataset to use ('synthetic', 'jester', 'movielens').
        n_runs (int): Number of runs.
        seed (Optional[int]): Random seed.

    Returns:
        Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]]: Results per dataset, budget, and agent.
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 10000, size=n_runs)
    results = {dataset: {horizon: {}}}

    if dataset == 'synthetic':
        envs = [DuelingBanditEnv.random_bt(k, d=5, seed=int(s)) for s in seeds]
    elif dataset == 'jester':
        ratings = load_jester_data(k=k, seed=seed)
        P = ratings_to_preference_matrix(ratings)
        envs = [DuelingBanditEnv(P, seed=int(s)) for s in seeds]
    elif dataset == 'movielens':
        ratings = load_movielens_data(k=k, seed=seed)
        P = ratings_to_preference_matrix(ratings)
        envs = [DuelingBanditEnv(P, seed=int(s)) for s in seeds]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    delta12 = np.mean([env.delta12() for env in envs])
    results[dataset]['separation'] = {'delta12': delta12}

    agents = {
        'Double TS': DoubleThompsonSamplingAgent,
        'Random': RandomPairAgent,
        'PARWiS': PARWiSAgent,
        'Contextual PARWiS': ContextualPARWiSAgent,
        'RL PARWiS': RLPARWiSAgent
    }

    for name, AgentClass in agents.items():
        metrics = {
            'mean_regret': np.zeros(horizon),
            'std_regret': np.zeros(horizon),
            'mean_recovery': np.zeros(horizon),
            'std_recovery': np.zeros(horizon),
            'mean_true_rank': np.zeros(horizon),
            'std_true_rank': np.zeros(horizon),
            'mean_reported_rank': np.zeros(horizon),
            'std_reported_rank': np.zeros(horizon)
        }
        for env, s in zip(envs, seeds):
            agent = AgentClass(k=k, d=5 if name == 'Contextual PARWiS' else 0, seed=int(s))
            run_results = run_simulation(env, agent, horizon)
            for key in metrics:
                metrics[key] += run_results[key] / n_runs
        results[dataset][horizon][name] = metrics

    return results