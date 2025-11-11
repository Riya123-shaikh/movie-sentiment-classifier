# movie-sentiment-classifier
NLP project that classifies movie reviews as positive or negative using NLTK and Scikit-learn.
# 🎬 Movie Review Sentiment Classifier

A Natural Language Processing (NLP) web app that predicts whether a movie review is **Positive 😊** or **Negative 😞** using NLTK and Scikit-learn.

## 🚀 Features
- Uses NLTK’s built-in `movie_reviews` dataset (2000 labeled reviews)
- TF-IDF vectorization for text features
- Logistic Regression model for classification
- Flask web interface for live sentiment prediction

## 🧠 Tech Stack
- Python
- NLTK
- Scikit-learn
- Flask
- Joblib

## 🏃‍♀️ How to Run
1. Clone this repo:
   ```bash
   git clone https://github.com/<your-username>/movie-sentiment-classifier.git
   cd movie-sentiment-classifier
2. install dependencies
pip install -r requirements.txt

3. Run the app
python app.py

4..Open your browser at http://127.0.0.1:5000

📊 Model Performance

Accuracy: 83.25% (on NLTK movie_reviews dataset)
