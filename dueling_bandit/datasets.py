import numpy as np
import pandas as pd
import os
from typing import Optional

def load_jester_data(k: int = 20, data_path: str = "data/jester_ratings.csv", seed: Optional[int] = None) -> np.ndarray:
    """
    Load Jester dataset and return a ratings matrix for k jokes.

    Args:
        k (int): Number of jokes to select.
        data_path (str): Path to Jester ratings CSV file.
        seed (Optional[int]): Random seed for reproducibility.

    Returns:
        np.ndarray: Ratings matrix (k x users) with NaN for missing values.
    """
    rng = np.random.default_rng(seed)
    if not os.path.exists(data_path):
        print(f"Warning: Jester dataset not found at {data_path}. Generating synthetic data.")
        return rng.uniform(-10, 10, size=(k, 1000))
    
    df = pd.read_csv(data_path)
    ratings = df.pivot(index='joke_id', columns='user_id', values='rating')
    num_ratings = ratings.count(axis=1)
    top_k_indices = num_ratings.nlargest(k).index
    ratings = ratings.loc[top_k_indices].to_numpy()
    return ratings

def load_movielens_data(k: int = 20, data_path: str = "data/ml-100k/u.data", seed: Optional[int] = None) -> np.ndarray:
    """
    Load MovieLens dataset and return a ratings matrix for k movies.

    Args:
        k (int): Number of movies to select.
        data_path (str): Path to MovieLens ratings file.
        seed (Optional[int]): Random seed for reproducibility.

    Returns:
        np.ndarray: Ratings matrix (k x users) with NaN for missing values.
    """
    rng = np.random.default_rng(seed)
    if not os.path.exists(data_path):
        print(f"Warning: MovieLens dataset not found at {data_path}. Generating synthetic data.")
        return rng.uniform(0.5, 5, size=(k, 1000))
    
    df = pd.read_csv(data_path, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    ratings = df.pivot(index='movie_id', columns='user_id', values='rating')
    num_ratings = ratings.count(axis=1)
    top_k_indices = num_ratings.nlargest(k).index
    ratings = ratings.loc[top_k_indices].to_numpy()
    return ratings

def ratings_to_preference_matrix(ratings: np.ndarray) -> np.ndarray:
    """
    Convert ratings matrix to pairwise preference matrix using logistic function.

    Args:
        ratings (np.ndarray): Ratings matrix (items x users).

    Returns:
        np.ndarray: Preference matrix (items x items).
    """
    mean_ratings = np.nanmean(ratings, axis=1)
    k = len(mean_ratings)
    P = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                diff = mean_ratings[i] - mean_ratings[j]
                P[i, j] = 1.0 / (1.0 + np.exp(-diff))
    np.fill_diagonal(P, 0.5)
    return P