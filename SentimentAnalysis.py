import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
import nltk
import pickle

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
train_data = pd.read_csv('/content/train.tsv', sep='\t')
test_data = pd.read_csv('/content/test.tsv', sep='\t')

# Function to preprocess text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub('<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphanumeric characters
    text = text.lower()  # Convert to lowercase
    return text

# Apply preprocessing to 'Phrase' column
train_data['Phrase'] = train_data['Phrase'].apply(preprocess_text)
test_data['Phrase'] = test_data['Phrase'].apply(preprocess_text)

# Remove stopwords using NLTK
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
train_data['Phrase'] = train_data['Phrase'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
test_data['Phrase'] = test_data['Phrase'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Lemmatization using NLTK
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
train_data['Phrase'] = train_data['Phrase'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
test_data['Phrase'] = test_data['Phrase'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# Encode labels
encoder = LabelEncoder()
train_data['Sentiment_encoded'] = encoder.fit_transform(train_data['Sentiment']) + 1  # Shift labels by 1

# Check if all classes are present
print(train_data['Sentiment_encoded'].value_counts())

# Hyperparameters
vocab_size = 3000
embedding_dim = 100
max_length = 200

# Tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(train_data['Phrase'])

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Convert sentences to sequences and pad sequences
train_sequences = tokenizer.texts_to_sequences(train_data['Phrase'])
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_sequences = tokenizer.texts_to_sequences(test_data['Phrase'])
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# One-hot encode labels for the neural network
onehot_encoder = OneHotEncoder(sparse=False)
train_labels = onehot_encoder.fit_transform(train_data['Sentiment_encoded'].values.reshape(-1, 1))

# Model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    Bidirectional(LSTM(64)),
    Dense(24, activation='relu'),
    Dense(5, activation='softmax')  # Ensure this matches the number of sentiment classes
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(train_padded, train_labels, epochs=3, validation_split=0.1, verbose=1)

# Save the model
model.save('sentiment_model.h5')

# Evaluate model on validation data
val_sequences = train_sequences[int(len(train_sequences) * 0.9):]
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding='post', truncating='post')
val_labels = train_data['Sentiment_encoded'].values[int(len(train_data) * 0.9):]

val_predictions = model.predict(val_padded)
val_pred_labels = np.argmax(val_predictions, axis=1) + 1  # Shift prediction by 1

accuracy = accuracy_score(val_labels, val_pred_labels)
print("Accuracy of prediction on validation set:", accuracy)
print("Classification Report:")
print(classification_report(val_labels, val_pred_labels))

# Function to predict sentiment for new reviews
def predict_sentiment(review, model, tokenizer, max_length):
    review = preprocess_text(review)
    review = ' '.join([word for word in review.split() if word not in stop_words])
    review = ' '.join([lemmatizer.lemmatize(word) for word in review.split()])
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)
    sentiment = np.argmax(prediction) + 1  # Shift prediction by 1
    return sentiment

# Load tokenizer and model for prediction
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = tf.keras.models.load_model('sentiment_model.h5')

# Test the prediction with different inputs
test_reviews = [
    "I absolutely loved this movie, it was fantastic!",  # Expecting positive sentiment
    "It was a terrible experience, I hated it.",         # Expecting negative sentiment
    "The plot was okay, but the acting was great.",      # Expecting neutral sentiment
    "One of the best films I've seen this year!",        # Expecting positive sentiment
    "It was not good, not bad, just average."            # Expecting neutral sentiment
]

for review in test_reviews:
    sentiment = predict_sentiment(review, model, tokenizer, max_length)
    print(f"Review: {review}")
    print(f"Predicted sentiment: {sentiment}")

# Plots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Confusion Matrix
cm = confusion_matrix(val_labels, val_pred_labels)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_+1, yticklabels=encoder.classes_+1)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ROC Curve for multi-class
# One-vs-Rest approach for multi-class ROC curve
from sklearn.metrics import roc_auc_score

fpr = {}
tpr = {}
roc_auc = {}

for i in range(5):
    fpr[i], tpr[i], _ = roc_curve(val_labels == (i+1), val_predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green']
for i, color in enumerate(colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i+1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Multi-class')
plt.legend(loc="lower right")
plt.show()

# Interactive input for sentiment prediction
def interactive_sentiment_analysis():
    while True:
        sentence = input("Enter a review (or type 'exit' to quit): ")
        if sentence.lower() == 'exit':
            break
        sentiment = predict_sentiment(sentence, model, tokenizer, max_length)
        print(f"Review: {sentence}")
        print(f"Predicted sentiment: {sentiment}")

# Start interactive sentiment analysis
interactive_sentiment_analysis()
