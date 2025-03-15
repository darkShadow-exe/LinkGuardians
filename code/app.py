from flask import Flask, request, jsonify
import requests
import pickle
from io import BytesIO

app = Flask(__name__)

# Function to load pickle files from GitHub
def load_pickle(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the download fails
    return pickle.load(BytesIO(response.content))

# Links for the model and vectorizer
model_link = 'https://github.com/KrishKumarRajani/LinkGuardians/raw/main/models/logistic_regression_model.pkl'
vectorizer_link = 'https://github.com/KrishKumarRajani/LinkGuardians/raw/main/models/tfidf_vectorizer.pkl'

# Load the model and vectorizer
print("Loading model and vectorizer...")
model = load_pickle(model_link)
vectorizer = load_pickle(vectorizer_link)
print("Model and vectorizer loaded successfully!")

@app.route('/')
def home():
    return "Welcome to LinkGuardians API! Use /predict endpoint to check URLs."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'url' not in data:
            return jsonify({'error': 'Missing URL'}), 400  # Handle missing input

        urls = [data['url']]  # Convert input to list
        urls_vectorised = vectorizer.transform(urls)  # Vectorize input URL
        result = model.predict(urls_vectorised)[0]  # Get prediction

        label = "Bad (Phishing)" if result == 'bad' else "Good (Safe)"
        return jsonify({'url': data['url'], 'prediction': label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Handle errors

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
