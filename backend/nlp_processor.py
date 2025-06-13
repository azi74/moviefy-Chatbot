import spacy
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
import json
import os
from config import Config

nltk.download('stopwords')
nltk.download('punkt')

class NLPProcessor:
    def __init__(self):
        # Load spaCy model
        self.nlp = spacy.load('en_core_web_md')
        
        # Load custom intents
        self.intents = self._load_intents()
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Load word embeddings (optional)
        # self.word_vectors = KeyedVectors.load_word2vec_format(Config.NLP_MODEL_PATH, binary=True)
    
    def _load_intents(self):
        """Load predefined intents from a JSON file"""
        intents_path = os.path.join(os.path.dirname(__file__), 'intents.json')
        with open(intents_path) as file:
            return json.load(file)
    
    def process_message(self, message):
        """Process user message and extract intent and entities"""
        doc = self.nlp(message.lower())
        
        # Extract entities
        entities = {
            'genres': self._extract_genres(doc),
            'actors': self._extract_people(doc, 'actor'),
            'directors': self._extract_people(doc, 'director'),
            'year_range': self._extract_year_range(doc),
            'rating': self._extract_rating(doc)
        }
        
        # Determine intent
        intent = self._classify_intent(message)
        
        return intent, entities
    
    def _extract_genres(self, doc):
        """Extract movie genres from text"""
        genres = []
        known_genres = ['action', 'comedy', 'drama', 'horror', 'sci-fi', 'romance',
                       'thriller', 'fantasy', 'animation', 'documentary']
        
        for token in doc:
            if token.text in known_genres:
                genres.append(token.text)
        
        return genres
    
    def _extract_people(self, doc, person_type):
        """Extract actors/directors using NER"""
        people = []
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                people.append(ent.text)
        return people
    
    def _extract_year_range(self, doc):
        """Extract year or year range"""
        years = []
        for ent in doc.ents:
            if ent.label_ == 'DATE' and ent.text.isdigit():
                year = int(ent.text)
                if 1900 <= year <= 2100:
                    years.append(year)
        
        if len(years) == 1:
            return (years[0] - 5, years[0] + 5)
        elif len(years) >= 2:
            return (min(years), max(years))
        return None
    
    def _extract_rating(self, doc):
        """Extract rating preferences"""
        for token in doc:
            if token.text in ['good', 'great', 'excellent']:
                return 7.5
            elif token.text in ['average', 'decent']:
                return 5.0
            elif token.text in ['high', 'top']:
                return 8.0
        return None
    
    def _classify_intent(self, message):
        """Classify user intent"""
        blob = TextBlob(message)
        
        # Simple rule-based classification
        if any(word in message.lower() for word in ['hi', 'hello', 'hey']):
            return 'greeting'
        elif any(word in message.lower() for word in ['recommend', 'suggest', 'movie', 'film']):
            return 'movie_recommendation'
        elif 'thank' in message.lower():
            return 'thanks'
        else:
            return 'unknown'