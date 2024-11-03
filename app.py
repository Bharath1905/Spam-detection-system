from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import os

app = Flask(__name__)

# Step 1: Load and preprocess data, train model, and save it
def train_and_save_model():
    # Load data
    data = pd.read_csv('spam.csv', encoding='latin-1')
    
    # Check and print column names
    print("Columns in CSV:", data.columns)

    # Adjusting to the actual column names from your CSV
    if 'v1' in data.columns and 'v2' in data.columns:
        data = data[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
    else:
        data = data.iloc[:, :2].rename(columns={data.columns[0]: 'label', data.columns[1]: 'message'})

    # Encode labels
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

    # Build pipeline
    model = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])

    # Train model
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'spam_classifier.pkl')
    print("Model trained and saved as 'spam_classifier.pkl'.")

# Check if model file exists; if not, train and save it
if os.path.exists('spam_classifier.pkl'):
    model = joblib.load('spam_classifier.pkl')
    print("Model loaded from 'spam_classifier.pkl'.")
else:
    print("Model file 'spam_classifier.pkl' not found. Training model...")
    train_and_save_model()
    model = joblib.load('spam_classifier.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        
        # Predict
        prediction = model.predict([message])[0]
        prediction_text = "Spam" if prediction == 1 else "Ham"
        
        return render_template('index.html', prediction_text=f'The message is: {prediction_text}', original_message=message)

if __name__ == "__main__":
    app.run(debug=True)
