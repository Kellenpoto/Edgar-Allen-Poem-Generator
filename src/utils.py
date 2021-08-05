import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, LSTM, Dense, Embedding

n_prev = 20
tokenizer = Tokenizer()

def load_text_data(filepath):
    data = pd.read_csv(filepath)
    text = data['text']
    return text

def tokenize_text(text):
    tokenizer.fit_on_texts(text)
    encoded_texts = tokenizer.texts_to_sequences(text)
    return encoded_texts

def _windowize_data(data, n_prev):
    data = np.array(data)
    n_predictions = len(data) - n_prev
    y = data[n_prev:]
    indices = np.arange(n_prev) + np.arange(n_predictions)[:, None]
    x = data[indices]
    return x, y

def _split_and_windowize(data, n_prev, fraction_test=0.3):
    n_predictions = len(data) - 2*n_prev
    
    n_test  = int(fraction_test * n_predictions)
    n_train = n_predictions - n_test   
    
    x_train, y_train = _windowize_data(data[:n_train], n_prev)
    x_test, y_test = _windowize_data(data[n_train:], n_prev)
    return x_train, x_test, y_train, y_test

def get_train_test_split(encoded_texts, n_prev):
    X_train, X_test, y_train, y_test = _split_and_windowize(encoded_texts[0], n_prev)
    for text in encoded_texts[1:]:
        temp_X_train, temp_X_test, temp_y_train, temp_y_test = _split_and_windowize(text,n_prev)
        X_train = np.concatenate((X_train, temp_X_train), axis=0)
        X_test = np.concatenate((X_test, temp_X_test), axis=0)
        y_train = np.concatenate((y_train, temp_y_train), axis=0)
        y_test = np.concatenate((y_test, temp_y_test), axis=0)
    return X_train, X_test, y_train, y_test

def create_model(num_words, n_prev):
    model = Sequential()
    model.add(Embedding(num_words, 128, input_length=n_prev))
    model.add(LSTM(128, input_shape=(n_prev,1), return_sequences=True))
    model.add(Dropout(.2))
    model.add(LSTM(128))
    model.add(Dropout(.2))
    model.add(Dense(num_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
    return model

if __name__ == '__main__':
    data = load_text_data('../data/preprocessed_data.csv')
    encoded_texts = tokenize_text(data)
    X_train, X_test, y_train, y_test = get_train_test_split(encoded_texts, n_prev)
    num_words = len(tokenizer.word_index) + 1
    model = create_model(num_words, n_prev)
