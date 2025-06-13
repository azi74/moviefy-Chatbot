import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'moviefy-secret-key')
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///moviefy.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # TMDB API
    TMDB_API_KEY = os.getenv('TMDB_API_KEY')
    TMDB_BASE_URL = 'https://api.themoviedb.org/3'
    
    # NLP Model Paths
    NLP_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../ml_models/saved_models/nlp_model')
    
    # Recommendation Model
    RECOMMENDER_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../ml_models/saved_models/recommender_model')