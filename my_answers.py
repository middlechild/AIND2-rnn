import numpy as np

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
import keras
import string

### DONE :: fill out the function below that transforms the input series 
### and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = [series[x:(x + window_size)] for x in range(series.size - window_size)]
    y = [series[x + window_size] for x in range(series.size - window_size)]
    
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:window_size])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y


### DONE :: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    # sequential model
    model = Sequential()
    
    # layer 1 uses an LSTM module with 5 hidden units
    model.add(LSTM(5, input_shape=(window_size,1)))
    
    # layer 2 uses a fully connected module with one unit
    model.add(Dense(1, activation='linear'))
    
    return model


### DONE :: return the text input with only ascii lowercase and the punctuation given below included
def cleaned_text(text):
    # characters to remove
    unwanted_characters = list(set(text) - set(string.ascii_letters) - set(['!', ',', '.', ':', ';', '?']))

    # remove unwanted characters from text
    for c in unwanted_characters:
        text = text.replace(c,' ').replace('  ',' ')
    
    return text


### DONE :: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = [text[x:(x + window_size)] for x in range(0, len(text) - window_size, step_size)]
    outputs = [text[x + window_size] for x in range(0, len(text) - window_size, step_size)]
    
    return inputs,outputs


# DONE :: build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    # sequential model
    model = Sequential()
    
    # layer 1 uses an LSTM module with 200 hidden units
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    
    # layer 2 uses a linear module where units equal the number of unique characters
    model.add(Dense(num_chars, activation='linear'))
    
    # layer 3 is an activation layer
    model.add(Activation("softmax"))

    return model
