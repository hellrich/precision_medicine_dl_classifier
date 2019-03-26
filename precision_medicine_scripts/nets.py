from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import * 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter, namedtuple
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from utils import *
import tensorflow.keras.backend as K
import tensorflow

#simple net (whole document as input)
def simple_embedding_net(pre_trained=False):
    embedding_in = Input(shape=(maxlen,), dtype='int32')
    if not pre_trained:
        embedding_layer = Embedding(input_dim=40000, output_dim=20, input_length=maxlen)(embedding_in)
    else:
        raise Exception("Not implemented")
    embedding_layer = Flatten()(embedding_layer)
    embedding_layer = Dropout(0.5)(embedding_layer)

    inputs = [Input(shape=(meshTags.shape[1],)), 
              Input(shape=(genes.shape[1],)), Input(shape=(organisms.shape[1],))]

    concatenated = Dropout(0.2)(keras.layers.concatenate([embedding_layer] + inputs))
    output = Dense(1, activation='sigmoid')(concatenated)

    model = keras.Model([embedding_in] + inputs, output)
    return model


#bonus info + sentence and document level lstms, see https://github.com/keras-team/keras/issues/5516#issuecomment-295016548
def two_level_lstm(embedding_weights, vocab_size, max_sentences, max_sentence_length, bonus_info_shapes=None, bi=False):
    def builder():
        e = make_embeddings(input_dim=vocab_size, input_length=max_sentence_length, weights=embedding_weights)
    
        # Encode each timestep
        bonus_info = [Input(shape=(x,)) for x in bonus_info_shapes] if bonus_info_shapes else None

        in_sentence = Input(shape=(max_sentence_length,), dtype='int64')
        embedded_sentence = e(in_sentence)
        if bi:
            lstm_sentence = Bidirectional(LSTM(64), merge_mode='sum')(embedded_sentence)
        else:
            lstm_sentence = LSTM(64)(embedded_sentence)
        encoded_model = keras.Model(in_sentence, lstm_sentence)

        sequence_input = Input(shape=(max_sentences, max_sentence_length), dtype='int64')
        seq_encoded = TimeDistributed(encoded_model)(sequence_input)
        seq_encoded = Dropout(0.2)(seq_encoded)

        # Encode entire sentence
        if bi:
            seq_encoded =  Dropout(0.2)(Bidirectional(LSTM(64), merge_mode='sum')(seq_encoded))
        else:
            seq_encoded =  Dropout(0.2)(LSTM(64)(seq_encoded))

        if bonus_info:
            seq_encoded = Dropout(0.5)(keras.layers.concatenate([seq_encoded] + bonus_info))

        # Prediction
        out_layer = Dense(1, activation='sigmoid')(seq_encoded)
        model = keras.Model([sequence_input] + bonus_info, out_layer)
        return model
    return builder

#bonus info + sentence and document level lstms, see https://github.com/keras-team/keras/issues/5516#issuecomment-295016548
def two_level_gru(embedding_weights, vocab_size, max_sentences, max_sentence_length, bonus_info_shapes=None, bi=False):
    def builder():
        e = make_embeddings(input_dim=vocab_size, input_length=max_sentence_length, weights=embedding_weights)
    
        # Encode each timestep
        bonus_info = [Input(shape=(x,)) for x in bonus_info_shapes] if bonus_info_shapes else None

        in_sentence = Input(shape=(max_sentence_length,), dtype='int64')
        embedded_sentence = e(in_sentence)
        if bi:
            lstm_sentence = Bidirectional(GRU(64), merge_mode='sum')(embedded_sentence)
        else:
            lstm_sentence = GRU(64)(embedded_sentence)
        encoded_model = keras.Model(in_sentence, lstm_sentence)

        sequence_input = Input(shape=(max_sentences, max_sentence_length), dtype='int64')
        seq_encoded = TimeDistributed(encoded_model)(sequence_input)
        seq_encoded = Dropout(0.2)(seq_encoded)

        # Encode entire sentence
        if bi:
            seq_encoded =  Dropout(0.2)(Bidirectional(GRU(64), merge_mode='sum')(seq_encoded))
        else:
            seq_encoded =  Dropout(0.2)(GRU(64)(seq_encoded))

        if bonus_info:
            seq_encoded = Dropout(0.5)(keras.layers.concatenate([seq_encoded] + bonus_info))

        # Prediction
        out_layer = Dense(1, activation='sigmoid')(seq_encoded)
        model = keras.Model([sequence_input] + bonus_info, out_layer)
        return model
    return builder

def two_level_lstm_attention(embedding_weights, vocab_size, max_sentences, max_sentence_length, bonus_info_shapes=None, bi=False):
    def builder():
        e = make_embeddings(input_dim=vocab_size, input_length=max_sentence_length, weights=embedding_weights)
    
        # Encode each timestep
        bonus_info = [Input(shape=(x,)) for x in bonus_info_shapes] if bonus_info_shapes else None

        in_sentence = Input(shape=(max_sentence_length,), dtype='int64')
        embedded_sentence = e(in_sentence)
        if bi:
            lstm_sentence = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded_sentence)
        else:
            lstm_sentence = LSTM(64, return_sequences=True)(embedded_sentence)
        lstm_sentence = AttentionWithContext(name="sentence_attention")(lstm_sentence)

        encoded_model = keras.Model(in_sentence, lstm_sentence)
        sequence_input = Input(shape=(max_sentences, max_sentence_length), dtype='int64')
        seq_encoded = TimeDistributed(encoded_model)(sequence_input)
        seq_encoded = Dropout(0.2)(seq_encoded)

        # Encode entire sentence
        if bi:
            seq_encoded =  Dropout(0.2)(Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(seq_encoded))
        else:
            seq_encoded =  Dropout(0.2)(LSTM(64, return_sequences=True)(seq_encoded))
        seq_encoded =  AttentionWithContext(name="document_attention")(seq_encoded)
        seq_encoded = Dropout(0.2)(seq_encoded)
        
        if bonus_info:
            seq_encoded = Dropout(0.5)(keras.layers.concatenate([seq_encoded] + bonus_info))

        # Prediction
        out_layer = Dense(1, activation='sigmoid')(seq_encoded)
        model = keras.Model([sequence_input] + bonus_info, out_layer)
        return model
    return builder

def two_level_gru_attention(embedding_weights, vocab_size, max_sentences, max_sentence_length, bonus_info_shapes=None, bi=False):
    def builder():
        e = make_embeddings(input_dim=vocab_size, input_length=max_sentence_length, weights=embedding_weights)
    
        # Encode each timestep
        bonus_info = [Input(shape=(x,)) for x in bonus_info_shapes] if bonus_info_shapes else None

        in_sentence = Input(shape=(max_sentence_length,), dtype='int64')
        embedded_sentence = e(in_sentence)
        if bi:
            lstm_sentence = Bidirectional(GRU(64, return_sequences=True), merge_mode='sum')(embedded_sentence)
        else:
            lstm_sentence = GRU(64, return_sequences=True)(embedded_sentence)
        lstm_sentence = AttentionWithContext(name="sentence_attention")(lstm_sentence)

        encoded_model = keras.Model(in_sentence, lstm_sentence)
        sequence_input = Input(shape=(max_sentences, max_sentence_length), dtype='int64')
        seq_encoded = TimeDistributed(encoded_model)(sequence_input)
        seq_encoded = Dropout(0.2)(seq_encoded)

        # Encode entire sentence
        if bi:
            seq_encoded =  Dropout(0.2)(Bidirectional(GRU(64, return_sequences=True), merge_mode='sum')(seq_encoded))
        else:
            seq_encoded =  Dropout(0.2)(GRU(64, return_sequences=True)(seq_encoded))
        seq_encoded =  AttentionWithContext(name="document_attention")(seq_encoded)
        seq_encoded = Dropout(0.2)(seq_encoded)
        
        if bonus_info:
            seq_encoded = Dropout(0.5)(keras.layers.concatenate([seq_encoded] + bonus_info))

        # Prediction
        out_layer = Dense(1, activation='sigmoid')(seq_encoded)
        model = keras.Model([sequence_input] + bonus_info, out_layer)
        return model
    return builder
    
#from https://gist.github.com/rmdort/596e75e864295365798836d9e8636033
class AttentionWithContext(Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self, init='glorot_uniform', kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None, bias_constraint=None,  **kwargs):
        self.supports_masking = True
        self.init = tensorflow.keras.initializers.get(init)
        self.kernel_initializer = tensorflow.keras.initializers.get('glorot_uniform')

        self.kernel_regularizer = tensorflow.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tensorflow.keras.regularizers.get(bias_regularizer)

        self.kernel_constraint = tensorflow.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tensorflow.keras.constraints.get(bias_constraint)

        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(int(input_shape[-1]), 1),
                                 initializer=self.kernel_initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.b = self.add_weight(shape=(int(input_shape[1]),),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)

        self.u = self.add_weight(shape=(int(input_shape[1]),),
                                 initializer=self.kernel_initializer,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.built = True

    def compute_mask(self, input, mask):
        return None

    def call(self, x, mask=None):
        # (x, 40, 300) x (300, 1)
        multData =  K.dot(x, self.kernel) # (x, 40, 1)
        multData = K.squeeze(multData, -1) # (x, 40)
        multData = multData + self.b # (x, 40) + (40,)

        multData = K.tanh(multData) # (x, 40)

        multData = multData * self.u # (x, 40) * (40, 1) => (x, 1)
        multData = K.exp(multData) # (X, 1)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            mask = K.cast(mask, K.floatx()) #(x, 40)
            multData = mask*multData #(x, 40) * (x, 40, )

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        multData /= K.cast(K.sum(multData, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        multData = K.expand_dims(multData)
        weighted_input = x * multData
        return K.sum(weighted_input, axis=1)


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1],)
