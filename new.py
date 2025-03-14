import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from flask import Flask, request, jsonify
from Crypto.Cipher import AES
import base64
import os

# Load dataset (replace 'emails.csv' with your dataset)
data = pd.read_csv('emails.csv')

# Preprocess data
data['text'] = data['text'].str.replace('[^\w\s]', '')
data['text'] = data['text'].str.lower()

# Feature extraction
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'Recall: {recall_score(y_test, y_pred)}')
print(f'F1-Score: {f1_score(y_test, y_pred)}')

# Encryption setup
encryption_key = os.urandom(32)  # Generate a random encryption key

def encrypt_text(text):
    cipher = AES.new(encryption_key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(text.encode())
    return base64.b64encode(cipher.nonce + tag + ciphertext).decode()

# Flask application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    email = request.json['email']
    email_processed = vectorizer.transform([email])
    prediction = model.predict(email_processed)
    result = 'phishing' if prediction[0] == 1 else 'legitimate'
    encrypted_email = encrypt_text(email)
    return jsonify({'result': result, 'encrypted_email': encrypted_email})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
