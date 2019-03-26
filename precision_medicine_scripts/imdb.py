from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
import pandas
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy 

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",num_words=None, skip_top=0, maxlen=666, seed=113, start_char=1,oov_char=2, index_from=3)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=666)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=666)

model = keras.Sequential()
model.add(Dense(500, input_dim=666, activation='relu'))
model.add(Dense(20, activation='tanh'))
model.add(Dense(1, name="output_layer", activation='sigmoid')) 

model.compile(loss='binary_crossentropy',
                      optimizer='AdaDelta',
                                    metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=50)
print(model.evaluate(x_test, y_test, verbose=1))
