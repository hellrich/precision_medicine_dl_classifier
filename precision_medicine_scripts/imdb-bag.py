from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Conv1D, Flatten, MaxPooling1D
import pandas
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np 

word_dimensions = 25000
embedding_dimensions = 50
max_input_length=250
(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",num_words=word_dimensions, skip_top=0, seed=113, start_char=1,oov_char=2, index_from=3)


def input2bag(array):
    def vector2bag(vector):
        bag = np.zeros(word_dimensions)
        for word_id in vector:
            bag[word_id] += 1
        return bag
    return np.asarray([vector2bag(row) for row in array])

x_train = input2bag(x_train)
x_test = input2bag(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)


def basic(model):
 model.add(Flatten()) #schlechter als lstm

def lstm(model):
 model.add(Bidirectional(LSTM(64))) #gut aber lahm

def conv(model):
 model.add(Conv1D(16, 4)) # gut und flott?
 model.add(Flatten())

def conv2(model):
    model.add( Conv1D(128, 5, activation='relu'))
    model.add( MaxPooling1D(5))
    model.add( Conv1D(128, 5, activation='relu'))
    model.add( MaxPooling1D(5))
    model.add( Conv1D(128, 5, activation='relu'))
    model.add( MaxPooling1D(5))  # global max pooling
    model.add( Flatten())
    model.add(Dense(128, activation='relu'))

model = keras.Sequential()
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                                    metrics=['accuracy'])
for i in range(10):
 model.fit(x_train, y_train, epochs=1, batch_size=50, verbose=0)
 print(i)
 print(model.evaluate(x_test,y_test, verbose=0))
