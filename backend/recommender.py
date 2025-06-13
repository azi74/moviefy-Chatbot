import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from config import Config
import pickle
import os
from tmdb.tmdb_client import TMDBClient

class MovieRecommender:
    def __init__(self):
        self.tmdb_client = TMDBClient()
        self.model = self._load_recommender_model()
        self.movie_data = self._load_movie_data()
        self._initialize_content_based_features()
    
    def _load_recommender_model(self):
        """Load trained recommendation model"""
        model_path = Config.RECOMMENDER_MODEL_PATH
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _load_movie_data(self):
        """Load or fetch movie data"""
        # In a real app, this would come from your database
        # For now, we'll use a sample dataset
        return pd.DataFrame(self.tmdb_client.get_popular_movies())
    
    def _initialize_content_based_features(self):
        """Prepare content-based features"""
        self.movie_data['combined_features'] = self.movie_data.apply(
            lambda x: f"{x['genres']} {x['keywords']} {x['cast']} {x['director']}", axis=1)
        
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.movie_data['combined_features'])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
    
    def recommend_movies(self, user_id=None, genres=None, actors=None, directors=None, year_range=None, rating=None):
        """Generate movie recommendations based on criteria"""
        # Filter movies based on criteria
        filtered_movies = self.movie_data.copy()
        
        if genres:
            filtered_movies = filtered_movies[
                filtered_movies['genres'].apply(lambda x: any(genre in x for genre in genres))
            ]
        
        if actors:
            filtered_movies = filtered_movies[
                filtered_movies['cast'].apply(lambda x: any(actor in x for actor in actors))
            ]
        
        if directors:
            filtered_movies = filtered_movies[
                filtered_movies['director'].apply(lambda x: any(director in x for director in directors))
            ]
        
        if year_range:
            filtered_movies = filtered_movies[
                (filtered_movies['year'] >= year_range[0]) & 
                (filtered_movies['year'] <= year_range[1])
            ]
        
        if rating:
            filtered_movies = filtered_movies[filtered_movies['vote_average'] >= rating]
        
        # If we have a collaborative filtering model
        if self.model and user_id:
            # Get top recommendations for user
            user_recommendations = []
            for movie_id in filtered_movies['id'].values:
                pred = self.model.predict(user_id, movie_id)
                user_recommendations.append((movie_id, pred.est))
            
            # Sort by predicted rating
            user_recommendations.sort(key=lambda x: x[1], reverse=True)
            top_movie_ids = [x[0] for x in user_recommendations[:10]]
            return filtered_movies[filtered_movies['id'].isin(top_movie_ids)].to_dict('records')
        
        # Fallback to content-based filtering
        if len(filtered_movies) > 0:
            # Get most similar movies based on content
            indices = pd.Series(filtered_movies.index, index=filtered_movies['id'])
            idx = indices[filtered_movies.iloc[0]['id']]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]  # Top 10 similar movies
            movie_indices = [i[0] for i in sim_scores]
            return filtered_movies.iloc[movie_indices].to_dict('records')
        
        # Fallback to popular movies if no filters match
        return self.movie_data.sort_values('popularity', ascending=False).head(10).to_dict('records')