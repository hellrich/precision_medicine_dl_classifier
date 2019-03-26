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
import train
import pickle
import json as j
import gzip
import glob
import sys

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
            return pickle.load(f)

    print("loading model")
    prepare_tensorflow()
    model = keras.models.load_model(folder+"/model.h5")

    toks = Tokenizers(load_tokenizer(folder+"/text"),
            load_tokenizer(folder+"/mesh"),
            load_tokenizer(folder+"/gene"),
            load_tokenizer(folder+"/organism"))

    return model, toks

Batch = namedtuple("Batch", ["text","mesh","gene","organism","pmids"])
def read_input(infolder, max_sentences = train.max_sentences, max_sentence_length = train.max_sentence_length):
    for file_name in glob.glob(infolder+"/*gz"):
        with gzip.open(file_name,"rt", encoding="utf-8", newline="\n") as f:
            for line in f:
                json = j.loads(line)
                mesh = " ".join(json["meshTags"]) if "meshTags" in json else " "
                genes = " ".join(json["genes"]) if "genes" in json else " "
                organisms = " ".join(json["organisms"]) if "organisms" in json else " "
                title = json["title"] if "title" in json else " "
                abstract = json["abstract"] if "abstract" in json else " "

                text = title.replace("[","").replace("]","") + " " + abstract
                text = text.lower().replace(". ", " . ").replace(", ", " , ").\
                                replace("? ", " ? ").replace("! ", " ! ")

                pmid = json["pubmedId"]
                yield text, mesh, genes, organisms, pmid

def process_input(model, toks, _input, sentence_wise=train.sentence_wise, batchsize=1000):

    def process_batch(batch):
        if batch:
            texts = sentence_processing(sentence_wise[0], sentence_wise[1], 
                batch.text, toks.text)
            mesh = toks.mesh.texts_to_matrix(batch.mesh, mode="binary")
            genes = toks.gene.texts_to_matrix(batch.gene, mode="binary")
            organisms = toks.organism.texts_to_matrix(batch.organism, mode="binary")
            return model.predict([texts, mesh, genes, organisms], batch_size=batchsize)
        return None

    print("processing")
    batch = Batch([],[],[],[],[])
    for text, mesh, gene, organism, pmid in _input:
        batch.text.append(text)
        batch.mesh.append(mesh)
        batch.gene.append(gene)
        batch.organism.append(organism)
        batch.pmids.append(pmid)
        if len(batch.text) >= batchsize:
            yield process_batch(batch), batch.pmids
            batch.text.clear()
            batch.mesh.clear()
            batch.gene.clear()
            batch.organism.clear()
            batch.pmids.clear()
    else:
        yield process_batch(batch), batch.pmids

if __name__ == "__main__":
    model, toks = load_model("model2017")
    with open("model2017_out", "w", 128000) as testout:
        for classification, pmids in process_input(model, toks, read_input("/data/data_corpora/SIGIR2019")):
            for c, pmid in zip(classification, pmids):
                print(c[0], pmid, file=testout)
