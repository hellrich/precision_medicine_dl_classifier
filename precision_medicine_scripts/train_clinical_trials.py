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

path_stub="/data/data_corpora/SIGIR2019_gsdata"
data2017 = TrialInputData(path_stub,"2017ct", "20180712processedGoldStandardCT2017.tsv.gz")
data2018 = TrialInputData(path_stub,"2018ct","20190111processedGoldStandardCT2018.tsv.gz")

bonus_info_max=10000
max_sentences = 30
max_sentence_length = 30
vocab_size=40000
max_epochs=8


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


def train_model(input_data, folder, attention=False, bi=False, gru=False):
    try:
        mkdir(folder)
    except OSError:
        pass
    all_sentences, all_mesh, all_genes, all_organisms, all_keywords, all_id, labels, word_index, toks = read_trial_data(input_data, vocab_size, max_sentences, max_sentence_length, bonus_info_max)
    embedding_weights = get_embedding_weights(input_dim=vocab_size, word_index=word_index)
    tuner = SmartTuner(4, {4: 0.5, 5: 4, 6: .7, 7: .7, 8: .7, 9: .7, 10: .7})
    bonus_info_shapes=(all_mesh.shape[1], all_genes.shape[1], all_organisms.shape[1], all_keywords.shape[1])
    if attention:
        if gru:
            model_provider = two_level_gru_attention(embedding_weights, vocab_size, max_sentences, 
                max_sentence_length, bonus_info_shapes=bonus_info_shapes, bi=bi)
        else:
            model_provider = two_level_lstm_attention(embedding_weights, vocab_size, max_sentences, 
                max_sentence_length, bonus_info_shapes=bonus_info_shapes)
    else:
        if gru:
            raise Exception("Not implemented")
        else:
            model_provider = two_level_lstm(embedding_weights, vocab_size, max_sentences, 
                max_sentence_length, bonus_info_shapes=bonus_info_shapes)

    model = train_and_return(model_provider, [all_sentences, all_mesh, all_genes, all_organisms, all_keywords], labels, 
        max_epochs=max_epochs, fine_tuner=tuner)

    model.save(folder+"/model.h5")
    store_tokenizer(folder+"/text", toks.text)
    store_tokenizer(folder+"/mesh", toks.mesh)
    store_tokenizer(folder+"/gene", toks.gene)
    store_tokenizer(folder+"/organism", toks.organism)
    store_tokenizer(folder+"/keywords", toks.keywords)
    
def store_tokenizer(name, tokenizer):
    with open(name+".pickle", 'wb') as f:
        print(name, len(tokenizer.word_index))
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

# loading
#with open('tokenizer.pickle', 'rb') as handle:
 #   tokenizer = pickle.load(handle)

if __name__ == "__main__":
    prepare_tensorflow()
    #train_model(data2017, "model2017",vocab_size=vocab_size, sentence_wise=sentence_wise, bonus_info_max=bonus_info_max)
    #train_model(data2017, "model2017_attention",vocab_size=vocab_size,sentence_wise=sentence_wise, bonus_info_max=bonus_info_max, attention=True)
    #train_model(data2017, "model2017_gru",vocab_size=vocab_size,sentence_wise=sentence_wise, bonus_info_max=bonus_info_max, attention=True, bi=True, gru=True)
    #train_model(data2017, "trial_model2017")
    #train_model(data2017, "trial_model2017_attention"attention=True)
    train_model(data2017, "trial_model2017_gru",attention=True, bi=True, gru=True)
    #train_model(data2018, "trial_model2018")
    #train_model(data2018, "trial_model2018_attention"attention=True)
    train_model(data2018, "trial_model2018_gru",attention=True, bi=True, gru=True)
