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

path_stub="/data/data_corpora/SIGIR2019_gsdata"
data2017 = TrialInputData(path_stub,"2017ct", "20180712processedGoldStandardCT2017.tsv.gz")
data2018 = TrialInputData(path_stub,"2018ct","20190111processedGoldStandardCT2018.tsv.gz")


  
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
max_sentences = 30
max_sentence_length = 30
vocab_size=40000
nfold=10
max_epochs=8

def run_and_report(tuners, model_provider, data, labels):
    for tuner in tuners:
        acc, accs = experiment(model_provider, data, labels, 
                   fine_tuner=tuner, max_epochs=max_epochs, nfold=nfold, verbose=0)
    print("Accuracy for", nfold, "crossvalidation:", "{:.3f}".format(acc))
    print("Raw values:",["{:.3f}".format(a) for a in accs])

def perform_cross_experiment(data1, data2):
    prepare_tensorflow()
    all_sentences, all_mesh, all_genes, all_organisms, all_keywords, all_id, labels, word_index, tokenizers = read_trial_data(data1, vocab_size, max_sentences, max_sentence_length, bonus_info_max)
    embedding_weights = get_embedding_weights(path="/home/hellrich/keras-test/embeddings/bio_nlp_win30_", input_dim=vocab_size, word_index=word_index)
    all_sentences2, all_mesh2, all_genes2, all_organisms2, all_keywords2, all_id2, labels2, word_index2 = read_trial_data(data2, vocab_size, max_sentences, max_sentence_length, bonus_info_max, tokenizers)

    tuners =[SmartTuner(4, {4: 0.5, 5: 4, 6: .7, 7: .7, 8: .7, 9: .7, 10: .7})]
    model_provider = two_level_gru(embedding_weights, vocab_size, max_sentences, max_sentence_length, bonus_info_shapes=(all_mesh.shape[1], all_genes.shape[1], all_organisms.shape[1], all_keywords.shape[1]), bi=True)
    for tuner in tuners:
        acc = cross_experiment(model_provider, [all_sentences, all_mesh, all_genes, all_organisms, all_keywords], labels,
        [all_sentences2, all_mesh2, all_genes2, all_organisms2, all_keywords2], labels2, fine_tuner=tuner, max_epochs=max_epochs)             
        print("Accuracy for cross-comparison:", "{:.3f}".format(acc))

def crossvalidation(input_data):
    prepare_tensorflow()
    #loading stuff
    all_sentences, all_mesh, all_genes, all_organisms, all_keywords, all_id, labels, word_index, tokenizers = read_trial_data(input_data, vocab_size, max_sentences, max_sentence_length, bonus_info_max)
    embedding_weights = get_embedding_weights(path="/home/hellrich/keras-test/embeddings/bio_nlp_win30_", input_dim=vocab_size, word_index=word_index)

    tuners =[
    SmartTuner(4, {4: 0.5, 5: 4, 6: .7, 7: .7, 8: .7, 9: .7, 10: .7}) 
            ]
    model_provider = two_level_gru(embedding_weights, vocab_size, max_sentences, max_sentence_length, bonus_info_shapes=(all_mesh.shape[1], all_genes.shape[1], all_organisms.shape[1], all_keywords.shape[1]), bi=True)

    run_and_report(tuners, model_provider, [all_sentences, all_mesh, all_genes, all_organisms, all_keywords], labels)

if __name__ == "__main__":
    print("2018 cross")
    crossvalidation(data2018)
    print("2017 cross")
    crossvalidation(data2017)
    print("2017 -> 2018")
    perform_cross_experiment(data2017,data2018)
    print("2018 -> 2017")
    perform_cross_experiment(data2018,data2017)
