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
import train_clinical_trials
import pickle
import json as j
import gzip
import glob
import sys
import nets
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

def load_model(folder):
    def load_tokenizer(name):
        with open(name+".pickle", 'rb') as f:
            t = pickle.load(f)
            #print(name, len(t.word_index))
            return t 

    print("loading model")
    model = keras.models.load_model(folder+"/model.h5", custom_objects={'AttentionWithContext':nets.AttentionWithContext})

    toks = TrialTokenizers(load_tokenizer(folder+"/text"),
            load_tokenizer(folder+"/mesh"),
            load_tokenizer(folder+"/gene"),
            load_tokenizer(folder+"/organism"),
            load_tokenizer(folder+"/keywords"))
    return model, toks

Batch = namedtuple("Batch", ["text","mesh","gene","organism","keywords","pmid"])
def process_input(model, toks, _input, max_sentences=train_clinical_trials.max_sentences, max_sentence_length=train_clinical_trials.max_sentence_length, batchsize=1000):

    def process_batch(batch):
        if batch:
            texts = sentence_processing(max_sentences, max_sentence_length,
                batch.text, toks.text, split=False)
            mesh = toks.mesh.texts_to_matrix(batch.mesh, mode="binary")
            genes = toks.gene.texts_to_matrix(batch.gene, mode="binary")
            organisms = toks.organism.texts_to_matrix(batch.organism, mode="binary")
            keywords = toks.keywords.texts_to_matrix(batch.keywords, mode="binary")
            return model.predict([texts, mesh, genes, organisms, keywords], batch_size=batchsize)
        return None

    print("processing")
    batch = Batch([],[],[],[],[],[])
    for text, mesh, gene, organism, keywords, pmid in _input:
        batch.text.append(text)
        batch.mesh.append(mesh)
        batch.gene.append(gene)
        batch.organism.append(organism)
        batch.keywords.append(keywords)
        batch.pmid.append(pmid)
        if len(batch.text) >= batchsize:
            yield process_batch(batch), batch.pmid
            batch.text.clear()
            batch.mesh.clear()
            batch.gene.clear()
            batch.organism.clear()
            batch.keywords.clear()
            batch.pmid.clear()
    else:
        yield process_batch(batch), batch.pmid

if __name__ == "__main__":
    in_file="/data/data_corpora/SIGIR2019_ct/clinicaltrials_json-h4.coling.uni-jena.de-1-0.json" 
    prepare_tensorflow()
    # model, toks = load_model("model2018_gru")
    # with open("/data/data_corpora/SIGIR2019_classified/model2018_gru_out", "w", 128000) as testout:
    #     for classification, pmids in process_input(model, toks, read_input("/data/data_corpora/SIGIR2019")):
    #         for c, pmid in zip(classification, pmids):
    #             print(c[0], pmid, file=testout)
    # model, toks = load_model("model2018_attention")
    # with open("/data/data_corpora/SIGIR2019_classified/model2018_attention_out", "w", 128000) as testout:
    #     for classification, pmids in process_input(model, toks, read_input("/data/data_corpora/SIGIR2019")):
    #         for c, pmid in zip(classification, pmids):
    #             print(c[0], pmid, file=testout)
    model, toks = load_model("trial_model2017_gru") # trial_model2018_gru
    with open("/data/data_corpora/SIGIR2019_classified/ct_model2017_out", "w", 128000) as testout:
        for classification, pmids in process_input(model, toks, read_trial_json_file(in_file)):
            for c, pmid in zip(classification, pmids):
                print(c[0], pmid, file=testout)
    model, toks = load_model("trial_model2018_gru") # trial_model2018_gru
    with open("/data/data_corpora/SIGIR2019_classified/ct_model2018_out", "w", 128000) as testout:
        for classification, pmids in process_input(model, toks, read_trial_json_file(in_file)):
            for c, pmid in zip(classification, pmids):
                print(c[0], pmid, file=testout)
