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

word_dimensions = 1000
embedding_dimensions = 50
max_input_length=250
(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",num_words=word_dimensions, skip_top=0, seed=113, start_char=1,oov_char=2, index_from=3)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_input_length)
y_train=np.array(y_train)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_input_length)
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

def id2word():
 word_to_id = keras.datasets.imdb.get_word_index()

 #only +2 cause original index starts at 1
 word_to_id = {k:(v+2) for k,v in word_to_id.items()}
 word_to_id["<PAD>"] = 0
 word_to_id["<START>"] = 1
 word_to_id["<UNK>"] = 2
 return {value:key for key,value in word_to_id.items()}

def load_embedding_text(dim):
 w2e = {}
 for line in open("glove.6B."+str(dim)+"d.txt"):
  line = line.strip().split(" ")
  w = line[0]
  e = [float(x) for x in line[1:]]
  w2e[w] = e
 return w2e

def embeddings(model):
 dim=50
 e = Embedding(word_dimensions, dim, input_length=max_input_length, trainable=False)
 model.add(e)

 id2word_map = id2word()
 weights = e.get_weights()[0]
 loaded = load_embedding_text(dim)
 for i in range(word_dimensions):
  w = id2word_map[i]
  if w in loaded:
   weights[i] = np.array(loaded[w])
 e.set_weights([weights])

model = keras.Sequential()
#embeddings(model)
model.add(Embedding(word_dimensions, embedding_dimensions, input_length=max_input_length))
basic(model)
#conv(model) #lstm(model)
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                                    metrics=['accuracy'])
for i in range(10):
 model.fit(x_train, y_train, epochs=1, batch_size=50, verbose=0)
 print(i)
 print(model.evaluate(x_test,y_test, verbose=0))
