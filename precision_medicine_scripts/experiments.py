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
nfold=10
max_epochs=8

def run_and_report(tuners, model_provider, data, labels):
    for tuner in tuners:
        acc, accs = experiment(model_provider, data, labels, 
                   fine_tuner=tuner, max_epochs=max_epochs, nfold=nfold)
    print("Accuracy for", nfold, "crossvalidation:", "{:.3f}".format(acc))
    print("Raw values:",["{:.3f}".format(a) for a in accs])

def lstm_experiment(input_data, bi=False, gru=False):
    prepare_tensorflow()
    #loading stuff
    texts, labels, word_index, meshTags, genes, organisms = read_data(input_data, vocab_size=vocab_size, 
                     sentence_wise=(max_sentences,max_sentence_length), bonus_info_max=bonus_info_max)
    embedding_weights = get_embedding_weights(path="/home/hellrich/keras-test/embeddings/bio_nlp_win30_", input_dim=vocab_size, word_index=word_index)

    tuners =[SmartTuner(4, {4: 0.5, 5: 4, 6: .7, 7: .7, 8: .7, 9: .7, 10: .7})]
    if gru:
        model_provider = two_level_gru(embedding_weights, vocab_size, max_sentences, 
            max_sentence_length, bonus_info_shapes=(meshTags.shape[1],genes.shape[1],organisms.shape[1]), bi=bi)
    else:
        model_provider = two_level_lstm(embedding_weights, vocab_size, max_sentences, 
            max_sentence_length, bonus_info_shapes=(meshTags.shape[1],genes.shape[1],organisms.shape[1]), bi=bi)
    run_and_report(tuners, model_provider, [texts, meshTags, genes, organisms], labels)

def cross_train_lstm(data1, data2, bi=False, gru=False):
    prepare_tensorflow()
    texts1, labels1, word_index1, meshTags1, genes1, organisms1, toks = read_data(data1, vocab_size=vocab_size, 
            sentence_wise=(max_sentences,max_sentence_length), bonus_info_max=bonus_info_max, return_tokenizers=True)
    embedding_weights = get_embedding_weights(path="/home/hellrich/keras-test/embeddings/bio_nlp_win30_",input_dim=vocab_size, word_index=word_index1)
    texts2, labels2, word_index2, meshTags2, genes2, organisms2 = read_data(data2, vocab_size=vocab_size, 
            sentence_wise=(max_sentences,max_sentence_length), bonus_info_max=bonus_info_max, existing_tokenizers=toks)
    tuners =[SmartTuner(4, {4: 0.5, 5: 4, 6: .7, 7: .7, 8: .7, 9: .7, 10: .7})]
    if gru:
        model_provider = two_level_gru(embedding_weights, vocab_size, max_sentences, 
            max_sentence_length, bonus_info_shapes=(meshTags1.shape[1],genes1.shape[1],organisms1.shape[1]), bi=bi)
    else:
        model_provider = two_level_lstm(embedding_weights, vocab_size, max_sentences, 
            max_sentence_length, bonus_info_shapes=(meshTags1.shape[1],genes1.shape[1],organisms1.shape[1]), bi=bi)
    for tuner in tuners:
        acc = cross_experiment(model_provider, [texts1, meshTags1, genes1, organisms1], labels1,
        [texts2, meshTags2, genes2, organisms2], labels2, fine_tuner=tuner, max_epochs=max_epochs)             
        print("Accuracy for cross-comparison:", "{:.3f}".format(acc))

def cross_train_lstm_attention(data1, data2, bi=False, gru=False):
    prepare_tensorflow()
    texts1, labels1, word_index1, meshTags1, genes1, organisms1, toks = read_data(data1, vocab_size=vocab_size, 
            sentence_wise=(max_sentences,max_sentence_length), bonus_info_max=bonus_info_max, return_tokenizers=True)
    embedding_weights = get_embedding_weights(path="/home/hellrich/keras-test/embeddings/bio_nlp_win30_",input_dim=vocab_size, word_index=word_index1)
    texts2, labels2, word_index2, meshTags2, genes2, organisms2 = read_data(data2, vocab_size=vocab_size, 
            sentence_wise=(max_sentences,max_sentence_length), bonus_info_max=bonus_info_max, existing_tokenizers=toks)
    tuners =[SmartTuner(4, {4: 0.5, 5: 4, 6: .7, 7: .7, 8: .7, 9: .7, 10: .7})]
    if gru:
        model_provider = two_level_gru_attention(embedding_weights, vocab_size, max_sentences, 
            max_sentence_length, bonus_info_shapes=(meshTags1.shape[1],genes1.shape[1],organisms1.shape[1]), bi=bi)
    else:
        model_provider = two_level_lstm_attention(embedding_weights, vocab_size, max_sentences, 
            max_sentence_length, bonus_info_shapes=(meshTags1.shape[1],genes1.shape[1],organisms1.shape[1]), bi=bi)
    for tuner in tuners:
        acc = cross_experiment(model_provider, [texts1, meshTags1, genes1, organisms1], labels1,
        [texts2, meshTags2, genes2, organisms2], labels2, fine_tuner=tuner, max_epochs=max_epochs)             
        print("Accuracy for cross-comparison:", "{:.3f}".format(acc))

def lstm_experiment_attention(input_data, bi=False, gru=False):
    prepare_tensorflow()
    #loading stuff
    texts, labels, word_index, meshTags, genes, organisms = read_data(input_data, vocab_size=vocab_size, 
                     sentence_wise=(max_sentences,max_sentence_length), bonus_info_max=bonus_info_max)
    embedding_weights = get_embedding_weights(path="/home/hellrich/keras-test/embeddings/bio_nlp_win30_", input_dim=vocab_size, word_index=word_index)

    tuners =[SmartTuner(4, {4: 0.5, 5: 4, 6: .7, 7: .7, 8: .7, 9: .7, 10: .7})]
    if gru:
        model_provider = two_level_gru_attention(embedding_weights, vocab_size, max_sentences, 
            max_sentence_length, bonus_info_shapes=(meshTags.shape[1],genes.shape[1],organisms.shape[1]), bi=bi)
    else:
        model_provider = two_level_lstm_attention(embedding_weights, vocab_size, max_sentences, 
            max_sentence_length, bonus_info_shapes=(meshTags.shape[1],genes.shape[1],organisms.shape[1]), bi=bi)
    run_and_report(tuners, model_provider, [texts, meshTags, genes, organisms], labels)



if __name__ == "__main__":
    #lstm_experiment(data2017) -> Accuracy for 10 crossvalidation: 0.770
    #   Raw values: ['0.771', '0.765', '0.778', '0.786', '0.756', '0.753', '0.758', '0.782', '0.781', '0.773']
    #lstm_experiment_less_info(data2017) -> Accuracy for 10 crossvalidation: 0.766
    #Raw values: ['0.768', '0.764', '0.769', '0.750', '0.750', '0.774', '0.768', '0.767', '0.786', '0.763']
    #lstm_experiment(data2018) -> Accuracy for 10 crossvalidation: 0.749
    #Raw values: ['0.743', '0.733', '0.747', '0.769', '0.749', '0.755', '0.742', '0.739', '0.750', '0.759']
    
    #cross_train_lstm(data2017, data2018) -> 0.675
    #cross_train_lstm(data2018, data2017) -> 0.681

    #meshminor bringt nix, wusste erik schon aus ablation
    #early stopping war nix
    #anderes sentence splitting, vermutlich egal, wenn gut auch in classify anpassen!
    #cross_train_lstm(data2018, data2017)  #0.688 statt .681 ; training 18, eval 17
    #print("\n","###"*10,"\n")
    #cross_train_lstm(data2017, data2018) #0.665 statt 0.675
    #print("\n","###"*10,"\n")
    #lstm_experiment(data2017) #0.773 ['0.772', '0.770', '0.772', '0.764', '0.754', '0.772', '0.792', '0.772', '0.792', '0.770']
    #print("\n","###"*10,"\n") 
    #lstm_experiment(data2018) #0.753 ['0.743', '0.757', '0.754', '0.763', '0.737', '0.752', '0.746', '0.757', '0.754', '0.766']

    #und noch mit 30er embeddings
    #cross_train_lstm(data2018, data2017)  #0.695 statt 0.688 ; training 18, eval 17
    #print("\n","###"*10,"\n")
    #cross_train_lstm(data2017, data2018) # 0.671 statt 0.665
    #print("\n","###"*10,"\n")
    #lstm_experiment(data2017) # 0.772 statt 0.773 
    #print("\n","###"*10,"\n") Raw values: ['0.773', '0.770', '0.783', '0.763', '0.758', '0.772', '0.778', '0.781', '0.778', '0.764']
    #lstm_experiment(data2018) # 0.755 statt 0.753
    #Raw values: ['0.760', '0.750', '0.761', '0.752', '0.772', '0.757', '0.732', '0.748', '0.757', '0.761']

    #attention muss auch noch...
   # lstm_experiment_attention(data2017) # 0.779 statt 0.772 
#Raw values: ['0.783', '0.772', '0.783', '0.767', '0.768', '0.776', '0.781', '0.797', '0.794', '0.768']
   # lstm_experiment_attention(data2018) # 0.755 statt 0.755 
    #Raw values: ['0.749', '0.758', '0.762', '0.766', '0.759', '0.757', '0.742', '0.741', '0.759', '0.760']
   # cross_train_lstm_attention(data2018, data2017)  # 0.685 statt 0.695 
   # print("\n","###"*10,"\n")
   # cross_train_lstm_attention(data2017, data2018) # 0.675 statt 0.671

    #attention mit dme vergessenem droput
    #lstm_experiment_attention(data2017) #0.775
   # Raw values: ['0.780', '0.784', '0.777', '0.763', '0.751', '0.772', '0.781', '0.788', '0.782', '0.766']
    #lstm_experiment_attention(data2018) #0.754
    #Raw values: ['0.741', '0.737', '0.763', '0.758', '0.769', '0.759', '0.752', '0.748', '0.754', '0.765']
    #cross_train_lstm_attention(data2018, data2017)  #0.681
    #cross_train_lstm_attention(data2017, data2018) # 0.671


    #BI-GRUs
    #und noch mit 30er embeddings
    #print("cross")
    #cross_train_lstm(data2018, data2017, bi=True, gru=True) # 0.671
    #cross_train_lstm(data2017, data2018, bi=True, gru=True) # 0.653
    
    #print("cross attention")
    #cross_train_lstm_attention(data2018, data2017,bi=True, gru=True)  #0.680
    #cross_train_lstm_attention(data2017, data2018,bi=True, gru=True) # 0.678

    #lstm_experiment(data2017,bi=True) # 0.772 statt 0.773 
    #lstm_experiment(data2018,bi=True) # 0.755 statt 0.753

    lstm_experiment_attention(data2017,bi=True, gru=True) #0.781
    #Raw values: ['0.777', '0.786', '0.781', '0.772', '0.777', '0.770', '0.786', '0.795', '0.799', '0.771']
    lstm_experiment_attention(data2018,bi=True, gru=True) #0.760
    #Raw values: ['0.754', '0.761', '0.752', '0.772', '0.770', '0.759', '0.755', '0.750', '0.762', '0.763']

