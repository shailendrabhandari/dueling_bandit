from typing import Dict, List
import numpy as np
from .environment import DuelingBanditEnv
from .agents import DoubleThompsonSamplingAgent, RandomPairAgent, PARWiSAgent, ContextualPARWiSAgent
from .datasets import load_jester_data, load_movielens_data

def run_simulation(env: DuelingBanditEnv, agent, horizon: int) -> Dict[str, List]:
    best_arm = env.best_arm()
    regret = []
    recovery = []
    true_rank_reported = []
    reported_rank_true = []
    cum_reg = 0

    if isinstance(agent, (PARWiSAgent, ContextualPARWiSAgent)):
        agent.phase1(env)
        t = env.k - 1
        recommendation = agent.recommend()
        true_order = env.true_order()
        regret.append(cum_reg)
        recovery.append(int(recommendation == best_arm))
        true_rank_reported.append(true_order.index(recommendation) + 1)
        if hasattr(agent, 'pi'):
            sorted_indices = np.argsort(-agent.pi)
            rank = np.where(sorted_indices == best_arm)[0][0] + 1
            reported_rank_true.append(rank)
        else:
            reported_rank_true.append(0)
    else:
        t = 0

    while t < horizon:
        a, b = agent.select_pair()
        winner, _ = env.duel(a, b)
        agent.update(a, b, winner)
        recommendation = agent.recommend()
        true_order = env.true_order()
        
        if winner != best_arm:
            cum_reg += 1
        regret.append(cum_reg)
        recovery.append(int(recommendation == best_arm))
        true_rank_reported.append(true_order.index(recommendation) + 1)
        if hasattr(agent, 'pi'):
            sorted_indices = np.argsort(-agent.pi)
            rank = np.where(sorted_indices == best_arm)[0][0] + 1
            reported_rank_true.append(rank)
        else:
            reported_rank_true.append(0)
        
        t += 1

    while len(regret) < horizon:
        regret.append(cum_reg)
        recovery.append(recovery[-1])
        true_rank_reported.append(true_rank_reported[-1])
        reported_rank_true.append(reported_rank_true[-1])

    return {
        'regret': regret[:horizon],
        'recovery': recovery[:horizon],
        'true_rank_reported': true_rank_reported[:horizon],
        'reported_rank_true': reported_rank_true[:horizon]
    }

def evaluate(k: int, horizon: int, dataset: str = 'synthetic', d: int = 0, n_runs: int = 30, seed_offset: int = 0) -> Dict[str, Dict[str, np.ndarray]]:
    results = {}
    separations = []
    agents = {
        'Double TS': DoubleThompsonSamplingAgent,
        'Random': RandomPairAgent,
        'PARWiS': PARWiSAgent,
        'Contextual PARWiS': ContextualPARWiSAgent
    }
    
    for name, AgentCls in agents.items():
        all_regrets = np.zeros((n_runs, horizon))
        all_recoveries = np.zeros((n_runs, horizon))
        all_true_ranks = np.zeros((n_runs, horizon))
        all_reported_ranks = np.zeros((n_runs, horizon))
        
        for run in range(n_runs):
            seed = seed_offset + run
            if dataset == 'synthetic':
                env = DuelingBanditEnv.random_bt(k, d, seed=seed)
            elif dataset == 'jester':
                ratings = load_jester_data(k=k)
                env = DuelingBanditEnv.from_ratings(ratings, k, seed=seed)
            elif dataset == 'movielens':
                ratings = load_movielens_data(k=k)
                env = DuelingBanditEnv.from_ratings(ratings, k, seed=seed)
            else:
                raise ValueError(f"Unknown dataset: {dataset}")

            if name == list(agents.keys())[0]:
                separations.append(env.separation())
            
            agent = AgentCls(k=k, d=d, seed=seed) if name == 'Contextual PARWiS' else AgentCls(k=k, seed=seed)
            sim_results = run_simulation(env, agent, horizon)
            
            all_regrets[run] = sim_results['regret']
            all_recoveries[run] = sim_results['recovery']
            all_true_ranks[run] = sim_results['true_rank_reported']
            all_reported_ranks[run] = sim_results['reported_rank_true']
        
        results[name] = {
            'mean_regret': all_regrets.mean(axis=0),
            'std_regret': all_regrets.std(axis=0),
            'mean_recovery': all_recoveries.mean(axis=0),
            'std_recovery': all_recoveries.std(axis=0),
            'mean_true_rank': all_true_ranks.mean(axis=0),
            'std_true_rank': all_true_ranks.std(axis=0),
            'mean_reported_rank': all_reported_ranks.mean(axis=0),
            'std_reported_rank': all_reported_ranks.std(axis=0)
        }
    
    results['separation'] = {
        'mean': np.mean(separations),
        'std': np.std(separations)
    }
    return results