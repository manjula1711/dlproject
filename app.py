from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model('model.h5')

# Load the CSV data and tokenizer
df = pd.read_csv('Top posts.csv')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['title'])

# Calculate the mean and standard deviation of scores
mean_score = df['score'].mean()
std_deviation = df['score'].std()

# Set the threshold based on statistics (e.g., mean + 1 * standard deviation)
threshold = mean_score + std_deviation

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        title = request.form['title']
        max_words = 10000  # Maximum number of words in your vocabulary
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(title)
        title_sequences = tokenizer.texts_to_sequences(title)
        max_title_length = max(len(seq) for seq in title_sequences)
        title_sequences = pad_sequences(title_sequences, maxlen=max_title_length)
        #title_sequence = tokenizer.texts_to_sequences([title])
        #title_sequence = pad_sequences(title_sequence, maxlen=max(len(seq) for seq in title_sequence))
        prediction = model.predict(title_sequences)

    
        if np.all(prediction >= threshold):
            result = "Good"
        else:
            result = "Bad"


        return render_template('result.html', title=title, prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
