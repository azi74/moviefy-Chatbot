# Moviefy Recommendation Chatbot 🎬🤖

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent movie recommendation chatbot that understands natural language queries and suggests personalized movie recommendations using hybrid recommendation algorithms.

![Chatbot Demo](demo.gif) *(Example GIF showing the chatbot in action)*

## Features ✨

- **Natural Language Understanding** 🗣️

  - Processes complex queries like "Show me funny horror movies from the 90s with Bruce Campbell"
  - Identifies genres, actors, directors, year ranges, and rating preferences
- **Hybrid Recommendation Engine** 🔍

  - Content-based filtering (TF-IDF + Cosine Similarity)
  - Collaborative filtering (KNN user-based)
  - Fallback to popularity-based recommendations
- **TMDB Integration** 🎥

  - Real-time movie data fetching
  - High-quality metadata including posters, cast, and ratings
- **Personalization** 👤

  - Learns from user interactions
  - Adapts recommendations based on feedback

## Tech Stack 🛠️

### Backend

| Component             | Technology                          |
| --------------------- | ----------------------------------- |
| Framework             | Python Flask                        |
| Natural Language      | spaCy, NLTK, TextBlob               |
| Recommendation Engine | scikit-learn, Surprise              |
| Database              | PostgreSQL (SQLite for development) |
| API Client            | Requests (TMDB API)                 |

### Machine Learning

| Model                   | Implementation                    |
| ----------------------- | --------------------------------- |
| Content-Based Filtering | TF-IDF Vectorization              |
| Collaborative Filtering | KNN User-Based (Surprise Library) |
| NLP Intent Detection    | Rule-Based + spaCy NER            |

## Installation 🚀

### Prerequisites

- Python 3.8+
- TMDB API key (free tier available)
- PostgreSQL (optional for development)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/moviefy-chatbot.git
   cd moviefy-chatbot
   ```

Create and activate virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install dependencies:

pip install -r requirements.txt
python -m spacy download en_core_web_md

Set up environment variables:
Create a `.env` file in the root directory:

TMDB_API_KEY=your_api_key_here
DATABASE_URL=postgresql://user:password@localhost/moviefy
SECRET_KEY=your_secret_key_here

Initialize database:

cd backend
python

>>> from app import db
>>> db.create_all()
>>> exit()
>>>
>>

Train recommendation models:

python ml_models/train_recommender.py

## Running the Application ▶️

Start the Flask development server:

python backend/app.py

The API will be available at `http://localhost:5000`

## PI Endpoints 📡

| Endpoint  | Method | Description            |
| --------- | ------ | ---------------------- |
| `/chat` | POST   | Main chatbot interface |

## Project Structure 🗂️

moviefy-chatbot/
├── backend/               # Flask application
│   ├── app.py             # Main application
│   ├── recommender.py      # Recommendation engine
│   ├── nlp_processor.py   # Natural language processing
│   ├── database/          # DB models and utilities
│   └── tmdb/              # TMDB API integration
├── ml_models/             # Machine learning components
├── tests/                 # Unit and integration tests
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables

## Contributing 🤝

We welcome contributions! Please follow these steps:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the [LICENSE](https://license/) file for details.

## Acknowledgments 🙏

* Data provided by [The Movie Database (TMDB)](https://www.themoviedb.org/)
* Inspired by Netflix/Amazon recommendation systems
* Built with ❤️ by asi
