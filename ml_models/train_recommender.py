import pandas as pd
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
import pickle
from backend.tmdb.tmdb_client import TMDBClient
from backend.config import Config

def prepare_training_data():
    """Prepare training data for collaborative filtering"""
    # In a real app, you would use actual user ratings from your database
    # Here we'll simulate some user ratings based on popularity
    
    tmdb_client = TMDBClient()
    movies = tmdb_client.get_popular_movies(limit=200)
    
    # Create sample user ratings
    ratings_data = []
    for user_id in range(1, 101):  # 100 simulated users
        for movie in movies[:50]:  # Each user rates 50 random movies
            # Simulate rating based on movie popularity with some noise
            base_rating = movie['vote_average'] / 2  # Scale to 0-5
            noise = (np.random.random() - 0.5) * 1.5  # Random noise
            rating = max(0.5, min(5.0, base_rating + noise))
            ratings_data.append({
                'user_id': user_id,
                'movie_id': movie['id'],
                'rating': rating
            })
    
    return pd.DataFrame(ratings_data)

def train_and_save_model():
    """Train and save the recommendation model"""
    # Prepare data
    ratings_df = prepare_training_data()
    
    # Define the rating scale
    reader = Reader(rating_scale=(0.5, 5.0))
    
    # Load the dataset
    data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)
    
    # Split data into train and test sets
    trainset, testset = train_test_split(data, test_size=0.2)
    
    # Train the model (user-based KNN)
    sim_options = {
        'name': 'cosine',
        'user_based': True
    }
    model = KNNBasic(sim_options=sim_options)
    model.fit(trainset)
    
    # Save the model
    model_path = Config.RECOMMENDER_MODEL_PATH
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model trained and saved to {model_path}")

if __name__ == '__main__':
    train_and_save_model()