import requests
from config import Config

class TMDBClient:
    def __init__(self):
        self.api_key = Config.TMDB_API_KEY
        self.base_url = Config.TMDB_BASE_URL
    
    def get_popular_movies(self, limit=100):
        """Fetch popular movies from TMDB"""
        url = f"{self.base_url}/movie/popular?api_key={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            movies = response.json().get('results', [])[:limit]
            return [self._format_movie(movie) for movie in movies]
        return []
    
    def get_movie_details(self, movie_id):
        """Get detailed information about a specific movie"""
        url = f"{self.base_url}/movie/{movie_id}?api_key={self.api_key}&append_to_response=credits,keywords"
        response = requests.get(url)
        if response.status_code == 200:
            return self._format_movie(response.json())
        return None
    
    def _format_movie(self, movie_data):
        """Format TMDB movie data into our schema"""
        return {
            'id': movie_data.get('id'),
            'title': movie_data.get('title'),
            'overview': movie_data.get('overview'),
            'genres': [g['name'] for g in movie_data.get('genres', [])],
            'year': int(movie_data.get('release_date', '1970').split('-')[0]) if movie_data.get('release_date') else 1970,
            'vote_average': movie_data.get('vote_average', 0),
            'popularity': movie_data.get('popularity', 0),
            'poster_path': f"https://image.tmdb.org/t/p/w500{movie_data.get('poster_path')}" if movie_data.get('poster_path') else None,
            'cast': [c['name'] for c in movie_data.get('credits', {}).get('cast', [])[:5]],
            'director': [c['name'] for c in movie_data.get('credits', {}).get('crew', []) if c['job'] == 'Director'][:2],
            'keywords': [k['name'] for k in movie_data.get('keywords', {}).get('keywords', [])]
        }