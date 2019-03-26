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
from nets import *
import pickle
from os import mkdir

data2017 = InputData("/home/hellrich/keras-test/precision_medicine_scripts/20180622processedGoldStandardTopics.tsv" ,
        "/home/hellrich/keras-test/precision_medicine_scripts/json-output-gs2017", True)
data2018 = InputData("/home/hellrich/keras-test/precision_medicine_scripts/20190111processedGoldStandardPub2018.tsv" ,
        "/home/hellrich/keras-test/precision_medicine_scripts/json-output-gs2018", False)

def prepare_tensorflow():
    ## extra imports to set GPU options
    import tensorflow as tf
     
    # TensorFlow wizardry
    config = tf.ConfigProto()
     
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
     
    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
     
    # Create a session with the above options specified.
    session = tf.Session(config=config)
    tf.keras.backend.set_session(session)

bonus_info_max=10000
max_sentences = 10
max_sentence_length = 30
sentence_wise=(max_sentences,max_sentence_length)
vocab_size=40000
nfold=10
max_epochs=8

def train_model(data, folder, vocab_size=vocab_size, 
            sentence_wise=sentence_wise, bonus_info_max=bonus_info_max, attention=False, bi=False, gru=False):
    try:
        mkdir(folder)
    except OSError:
        pass
    texts, labels, word_index, meshTags, genes, organisms, toks = read_data(data, vocab_size=vocab_size, 
            sentence_wise=sentence_wise, bonus_info_max=bonus_info_max, return_tokenizers=True)
    embedding_weights = get_embedding_weights(input_dim=vocab_size, word_index=word_index)
    tuner = SmartTuner(4, {4: 0.5, 5: 4, 6: .7, 7: .7, 8: .7, 9: .7, 10: .7})
    if attention:
        if gru:
            model_provider = two_level_gru_attention(embedding_weights, vocab_size, max_sentences, 
                max_sentence_length, bonus_info_shapes=(meshTags.shape[1],genes.shape[1],organisms.shape[1]), bi=bi)
        else:
            model_provider = two_level_lstm_attention(embedding_weights, vocab_size, max_sentences, 
                max_sentence_length, bonus_info_shapes=(meshTags.shape[1],genes.shape[1],organisms.shape[1]))
    else:
        if gru:
            raise Exception("Not implemented")
        else:
            model_provider = two_level_lstm(embedding_weights, vocab_size, max_sentences, 
                max_sentence_length, bonus_info_shapes=(meshTags.shape[1],genes.shape[1],organisms.shape[1]))

    model = train_and_return(model_provider, [texts, meshTags, genes, organisms], labels, 
        max_epochs=max_epochs, fine_tuner=tuner)
    model.save(folder+"/model.h5")
    store_tokenizer(folder+"/text", toks.text)
    store_tokenizer(folder+"/mesh", toks.mesh)
    store_tokenizer(folder+"/gene", toks.gene)
    store_tokenizer(folder+"/organism", toks.organism)
    
def store_tokenizer(name, tokenizer):
    with open(name+".pickle", 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

# loading
#with open('tokenizer.pickle', 'rb') as handle:
 #   tokenizer = pickle.load(handle)

if __name__ == "__main__":
    prepare_tensorflow()
    #train_model(data2017, "model2017",vocab_size=vocab_size, sentence_wise=sentence_wise, bonus_info_max=bonus_info_max)
    #train_model(data2017, "model2017_attention",vocab_size=vocab_size,sentence_wise=sentence_wise, bonus_info_max=bonus_info_max, attention=True)
    #train_model(data2017, "model2017_gru",vocab_size=vocab_size,sentence_wise=sentence_wise, bonus_info_max=bonus_info_max, attention=True, bi=True, gru=True)
    train_model(data2018, "model2018",vocab_size=vocab_size, sentence_wise=sentence_wise, bonus_info_max=bonus_info_max)
    train_model(data2018, "model2018_attention",vocab_size=vocab_size,sentence_wise=sentence_wise, bonus_info_max=bonus_info_max, attention=True)
    train_model(data2018, "model2018_gru",vocab_size=vocab_size,sentence_wise=sentence_wise, bonus_info_max=bonus_info_max, attention=True, bi=True, gru=True)