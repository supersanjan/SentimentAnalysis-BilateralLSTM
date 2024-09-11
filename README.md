Sentiment Analysis on Movie Reviews

This repository contains a sentiment analysis project focused on classifying movie reviews into multiple sentiment categories using NLP and deep learning techniques. The dataset used consists of movie reviews labeled with sentiments on a scale of 1 to 5, where each number represents a different sentiment intensity.

Project Overview

This project uses a combination of text preprocessing, tokenization, and a Bidirectional LSTM neural network to classify the sentiment of movie reviews. The model is trained to recognize the sentiment behind textual reviews and can predict the sentiment of new reviews after training.

Model Architecture
- Embedding Layer: Converts text into dense vectors.
- Bidirectional LSTM Layer: Captures dependencies in text data in both forward and backward directions.
- Dense Layer: Includes a fully connected layer with ReLU activation.
- Output Layer: Produces a probability distribution for the five sentiment classes using Softmax activation.

Dataset

The dataset is composed of two files:

- `train.tsv`: Contains movie reviews and their associated sentiment labels.
- `test.tsv`: Contains movie reviews without labels (for inference purposes).

Both datasets are in TSV (tab-separated values) format.

File Structure

```
- Datasets/
  - train.tsv
  - test.tsv
- SentimentAnalysis.py
- tokenizer.pickle
- sentiment_model.h5
- README.md
```

Requirements

To run this project, the following Python libraries are required:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- nltk
- pickle

You can install them by running:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow nltk
```

Preprocessing

Before feeding the text into the model, it goes through the following preprocessing steps:

1. HTML tag removal
2. URL removal
3. Lowercasing
4. Removing non-alphabetic characters
5. Stopword removal using NLTK’s stopwords list
6. Lemmatization using NLTK’s WordNet Lemmatizer

Model Training

- The tokenizer is fitted on the training data and saved as `tokenizer.pickle`.
- Reviews are tokenized and padded to ensure uniform input length (`max_length = 200`).
- The model is trained for 10 epochs with a validation split of 0.1.

```python
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    Bidirectional(LSTM(64)),
    Dense(24, activation='relu'),
    Dense(5, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

The model is saved as `sentiment_model.h5`.

Evaluation

- Accuracy and classification reports are generated for the validation data.
- A confusion matrix and ROC curves for multi-class classification are plotted.

Accuracy Example

```python
accuracy = accuracy_score(val_labels, val_pred_labels)
print("Accuracy of prediction on validation set:", accuracy)
```

Confusion Matrix Plot

```python
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```

ROC Curve Plot

```python
for i, color in enumerate(colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i+1}')
```

Prediction on New Reviews

You can use the model to predict the sentiment of new movie reviews by using the `predict_sentiment` function. Example:

```python
test_reviews = [
    "I absolutely loved this movie, it was fantastic!",
    "It was a terrible experience, I hated it."
]

for review in test_reviews:
    sentiment = predict_sentiment(review, model, tokenizer, max_length)
    print(f"Review: {review}")
    print(f"Predicted sentiment: {sentiment}")
```

Interactive Sentiment Analysis

You can interactively input reviews and get the predicted sentiment using the following function:

```python
interactive_sentiment_analysis()
```

Type `exit` to quit the interactive session.

 Plots

During training, the project plots:

- Training and Validation Accuracy**
- Training and Validation Loss**
- Confusion Matrix**
- ROC Curve for Multi-class Sentiment**

How to Run

1. Clone the repository.
2. Ensure the required packages are installed.
3. Run the script `SentimentAnalysis.py`.
4. Use the interactive sentiment analysis function to test the model on new reviews.

Future Improvements

- Fine-tune the model hyperparameters for better accuracy.
- Experiment with other deep learning models like GRU or CNNs for text classification.
- Use a more comprehensive dataset for training.

