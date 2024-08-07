from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
import json

app = Flask(__name__)
CORS(app)


# Load the model
model = load_model('mental_model.h5')

# Load the tokenizer and label encoder
with open('projectone.json') as json_file:
    intents_data = json.load(json_file)

patterns = [pattern for intent in intents_data['intents'] for pattern in intent['patterns']]
tags = [intent.get('tag', None) for intent in intents_data['intents']]

# Tokenize patterns and prepare encoder
tokenizer = Tokenizer(oov_token="<OOV>")  
tokenizer.fit_on_texts(patterns)

label_encoder = LabelEncoder()
label_encoder.fit(tags)

def preprocess_input(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=18, padding='post')
    return padded_sequence

@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.json
    user_input = request_data.get('message')
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    padded_sequence = preprocess_input(user_input)
    prediction = model.predict(padded_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    confidence = np.max(prediction)

    confidence_threshold = 0.6

    if confidence < confidence_threshold:
        reply = "I'm sorry, I didn't understand that. Could you please rephrase?"
    else:
        # Find the response associated with the predicted label
        for intent in intents_data['intents']:
            if intent['tag'] == predicted_label[0]:
                reply = np.random.choice(intent['responses'])
                break

    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(port=10000)
