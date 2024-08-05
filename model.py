import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

# Load the dataset
with open('projectone.json') as json_file:
    data = json.load(json_file)

# Extract patterns and tags
patterns = []
tags = []
question_ids = []  # New list to store question IDs

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent.get('tag', None))  # Handle potential missing 'tag' key
        # Handle the case where 'Question_ID' might be missing
        question_id = intent.get('Question_ID', None)
        question_ids.append(question_id)

# Encode the tags
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(tags)

# Tokenize the patterns
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(patterns)
sequences = tokenizer.texts_to_sequences(patterns)
padded_sequences = pad_sequences(sequences, padding='post')

# Get the vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# Get the number of classes
num_classes = len(label_encoder.classes_)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, 16, input_length=padded_sequences.shape[1]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Split the data into training and validation sets
training_size = int(len(padded_sequences) * 0.8)
x_train = padded_sequences[:training_size]
y_train = labels[:training_size]
x_val = padded_sequences[training_size:]
y_val = labels[training_size:]

# Train the model
history = model.fit(x_train, y_train, epochs=800, validation_data=(x_val, y_val))

model.save("mental_model.h5")

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=2)
print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
print(f"Validation loss: {val_loss:.2f}")

# Calculate precision, recall, and F1-score
# Generate predictions on the validation set
y_pred_probs = model.predict(x_val)

# Convert probabilities to class labels
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Find unique classes in predictions and validation set
unique_predicted = np.unique(y_pred_classes)
unique_val = np.unique(y_val)

# Get the target names for the classes present in both
common_classes = np.intersect1d(unique_predicted, unique_val)
target_names = label_encoder.classes_[common_classes]

# Generate classification report, only for classes present in both sets
report = classification_report(
    y_val,
    y_pred_classes,
    labels=common_classes,  # Use only common classes
    target_names=target_names
)
print(report)

# Chatbot running section


# Function to preprocess user input
def preprocess_input(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=18, padding='post')
    return padded_sequence


# Function to generate chatbot response
def generate_response(text):
    padded_sequence = preprocess_input(text)
    prediction = model.predict(padded_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    confidence = np.max(prediction)

    # Set a confidence threshold
    confidence_threshold = 0.6

    if confidence < confidence_threshold:
        return "I'm sorry, I didn't understand that. Could you please rephrase?"
    else:
        for intent in data['intents']:
            if intent['tag'] == predicted_label:
                return np.random.choice(intent['responses'])


# Run the chatbot
print("Start chatting with mental health chatbot (type 'quit' to stop)!")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    response = generate_response(user_input)
    print(f"Bot: {response}")
