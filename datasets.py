import numpy as np
import pandas as pd

def load_jester_data(k: int = 20) -> np.ndarray:
    """
    Load Jester dataset and return a ratings matrix for the top k jokes.
    Ratings are from -10 to +10, with NaN for missing values.
    Returns: ratings matrix (k x users)
    """
    print("Simulating Jester dataset loading (replace with actual data loading)...")
    rng = np.random.default_rng(42)
    num_users = 1000
    num_jokes = 100
    ratings = rng.uniform(-10, 10, size=(num_jokes, num_users))
    mask = rng.choice([0, 1], size=(num_jokes, num_users), p=[0.7, 0.3])
    ratings[mask == 0] = np.nan
    num_ratings = np.sum(~np.isnan(ratings), axis=1)
    top_k_indices = np.argsort(-num_ratings)[:k]
    return ratings[top_k_indices]

def load_movielens_data(k: int = 20) -> np.ndarray:
    """
    Load MovieLens dataset and return a ratings matrix for the top k movies.
    Ratings are from 1 to 5, with missing values as NaN.
    Returns: ratings matrix (k x users)
    """
    print("Simulating MovieLens dataset loading (replace with actual data loading)...")
    rng = np.random.default_rng(42)
    num_users = 943
    num_movies = 1682
    ratings = rng.uniform(1, 5, size=(num_movies, num_users))
    mask = rng.choice([0, 1], size=(num_movies, num_users), p=[0.9, 0.1])
    ratings[mask == 0] = np.nan
    num_ratings = np.sum(~np.isnan(ratings), axis=1)
    top_k_indices = np.argsort(-num_ratings)[:k]
    return ratings[top_k_indices]