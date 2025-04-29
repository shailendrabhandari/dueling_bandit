import numpy as np
import pandas as pd
import os

def load_jester_data(k: int = 20, data_path: str = "jester_ratings.csv") -> np.ndarray:
    """
    Load Jester dataset and return a ratings matrix for the top k jokes.
    
    Args:
        k (int): Number of jokes to select.
        data_path (str): Path to Jester ratings CSV file.
    
    Returns:
        np.ndarray: Ratings matrix (k x users) with NaN for missing values.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Jester dataset not found at {data_path}. Download from http://eigentaste.berkeley.edu/dataset/")
    
    # Load CSV (assuming columns: user_id, joke_id, rating)
    df = pd.read_csv(data_path)
    ratings = df.pivot(index='joke_id', columns='user_id', values='rating')
    
    # Select top k jokes by number of ratings
    num_ratings = ratings.count(axis=1)
    top_k_indices = num_ratings.nlargest(k).index
    ratings = ratings.loc[top_k_indices].to_numpy()
    
    return ratings

def load_movielens_data(k: int = 20, data_path: str = "ml-100k/u.data") -> np.ndarray:
    """
    Load MovieLens dataset and return a ratings matrix for the top k movies.
    
    Args:
        k (int): Number of movies to select.
        data_path (str): Path to MovieLens ratings file.
    
    Returns:
        np.ndarray: Ratings matrix (k x users) with NaN for missing values.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"MovieLens dataset not found at {data_path}. Download from https://grouplens.org/datasets/movielens/")
    
    # Load MovieLens 100K data (format: user_id, movie_id, rating, timestamp)
    df = pd.read_csv(data_path, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    ratings = df.pivot(index='movie_id', columns='user_id', values='rating')
    
    # Select top k movies by number of ratings
    num_ratings = ratings.count(axis=1)
    top_k_indices = num_ratings.nlargest(k).index
    ratings = ratings.loc[top_k_indices].to_numpy()
    
    return ratings
