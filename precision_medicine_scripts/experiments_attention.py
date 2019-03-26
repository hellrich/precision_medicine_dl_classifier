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
vocab_size=40000
nfold=1
max_epochs=8

def run_and_report(tuners, model_provider, data, labels):
    for tuner in tuners:
        acc, accs = experiment(model_provider, data, labels, 
                   fine_tuner=tuner, max_epochs=max_epochs, nfold=nfold, verbose=1)
    print("Accuracy for", nfold, "crossvalidation:", "{:.3f}".format(acc))
    print("Raw values:",["{:.3f}".format(a) for a in accs])

def cross_train_lstm_attention(data1, data2):
    prepare_tensorflow()
    texts1, labels1, word_index1, meshTags1, genes1, organisms1, toks = read_data(data1, vocab_size=vocab_size, 
            sentence_wise=(max_sentences,max_sentence_length), bonus_info_max=bonus_info_max, return_tokenizers=True)
    embedding_weights = get_embedding_weights(path="/home/hellrich/keras-test/embeddings/bio_nlp_win30_",input_dim=vocab_size, word_index=word_index1)
    texts2, labels2, word_index2, meshTags2, genes2, organisms2 = read_data(data2, vocab_size=vocab_size, 
            sentence_wise=(max_sentences,max_sentence_length), bonus_info_max=bonus_info_max, existing_tokenizers=toks)
    tuners =[SmartTuner(4, {4: 0.5, 5: 4, 6: .7, 7: .7, 8: .7, 9: .7, 10: .7})]
    model_provider = two_level_lstm_attention(embedding_weights, vocab_size, max_sentences, 
        max_sentence_length, bonus_info_shapes=(meshTags1.shape[1],genes1.shape[1],organisms1.shape[1]))
    for tuner in tuners:
        acc = cross_experiment(model_provider, [texts1, meshTags1, genes1, organisms1], labels1,
        [texts2, meshTags2, genes2, organisms2], labels2, fine_tuner=tuner, max_epochs=max_epochs)             
        print("Accuracy for cross-comparison:", "{:.3f}".format(acc))

def lstm_experiment_attention(input_data):
    prepare_tensorflow()
    #loading stuff
    texts, labels, word_index, meshTags, genes, organisms = read_data(input_data, vocab_size=vocab_size, 
                     sentence_wise=(max_sentences,max_sentence_length), bonus_info_max=bonus_info_max)
    embedding_weights = get_embedding_weights(path="/home/hellrich/keras-test/embeddings/bio_nlp_win30_", input_dim=vocab_size, word_index=word_index)

    tuners =[SmartTuner(0, {}), 
    SmartTuner(4, {4: 0.5, 5: 4, 6: .7, 7: .7, 8: .7, 9: .7, 10: .7}) 
            ]
    model_provider = two_level_lstm_attention(embedding_weights, vocab_size, max_sentences, 
        max_sentence_length, bonus_info_shapes=(meshTags.shape[1],genes.shape[1],organisms.shape[1]))
    run_and_report(tuners, model_provider, [texts, meshTags, genes, organisms], labels)



if __name__ == "__main__":
    lstm_experiment_attention(data2018)
    lstm_experiment_attention(data2017)
    
