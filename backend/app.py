from flask import Flask, request, jsonify
from flask_cors import CORS
from nlp_processor import NLPProcessor
from recommender import MovieRecommender
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Initialize components
nlp_processor = NLPProcessor()
recommender = MovieRecommender()

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        user_id = data.get('user_id', 'anonymous')  # For personalization
        
        # Process the message with NLP
        intent, entities = nlp_processor.process_message(user_message)
        
        # Generate response based on intent
        if intent == 'greeting':
            response = {
                "message": "Hi! I'm your Moviefy assistant. I can recommend movies based on your preferences. Tell me what kind of movies you like!",
                "type": "text"
            }
        elif intent == 'movie_recommendation':
            # Get recommendations based on entities extracted
            recommendations = recommender.recommend_movies(
                user_id=user_id,
                genres=entities.get('genres', []),
                actors=entities.get('actors', []),
                directors=entities.get('directors', []),
                year_range=entities.get('year_range', None),
                rating=entities.get('rating', None)
            )
            
            response = {
                "message": "Here are some movies you might like:",
                "type": "recommendations",
                "data": recommendations
            }
        else:
            response = {
                "message": "I'm not sure I understand. Could you tell me more about what kind of movies you like?",
                "type": "text"
            }
            
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)