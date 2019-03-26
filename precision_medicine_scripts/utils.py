from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import * 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter, namedtuple
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


from enum import Enum
from os import listdir
import json as j
from collections import namedtuple
import gzip
import glob

#non-error prone training?
def get_embedding_weights(input_dim, word_index, path="/home/hellrich/keras-test/embeddings/bio_nlp_win2_", suffix="d.vec",
               embedding_dim=200):
    def load_embedding_text(dim):
        w2e = {}
        for line in open(path+str(dim)+suffix):
            line = line.strip().split(" ")
            if len(line) > 2:
                w = line[0]
                e = np.asarray(line[1:], dtype="float32")
                w2e[w] = e
        return w2e
    
    index2word = {v : k for k, v in word_index.items()}
    weights = np.zeros((input_dim, embedding_dim))
    loaded = load_embedding_text(embedding_dim)
    entries = 0
    for i in range(1,input_dim):
        w = index2word[i]
        if w in loaded:
            weights[i] = np.array(loaded[w])
            entries += 1
    return weights

def make_embeddings(input_dim,input_length,weights,embedding_dim=200,trainable=False):
    _weights = [np.copy(x) for x in weights]
    return Embedding(input_dim, embedding_dim, weights=[weights],
                      input_length=input_length, trainable=trainable) #input_dim + 1 ?

BonusInfo = namedtuple('BonusInfo', ["meshTags","genes", "organisms"])

def parse_json(path):
    pmid2info = {}
    for json_file in [x for x in listdir(path) if x.endswith(".json")]:
        with open(path+"/"+json_file, "r", encoding="utf8") as open_json_file:
            json = j.load(open_json_file)
            if "meshTags" in json:
                meshTags = " ".join(json["meshTags"])
            else:
                meshTags = " "
            if "genes" in json:
                genes = " ".join(json["genes"])
            else:
                genes = " "
            if "organisms" in json:
                organisms = " ".join(json["organisms"])
            else:
                organisms = " "
            #abstract = json["abstract"]
            #title = json["title"]
            pmid = json["pubmedId"]
            pmid2info[pmid] = BonusInfo(meshTags, genes, organisms)
    return pmid2info

TrialInputData = namedtuple("TrialInputData", ["folder","json","tsv"])
def read_trial_tsv(input_data):
    id2pm = {}
    with gzip.open(input_data.folder+"/"+input_data.tsv,"rt", encoding="utf-8", newline="\n") as f:
        next(f) #discard first line
        for line in f:
            line = line.split("\t")
            _id = line[2]
            pm = line[3]
            if _id in id2pm and id2pm[_id]:
                pass #einmal PM, immer PM
            elif pm == 'Human PM' or pm == 'Animal PM':
                id2pm[_id] = True 
            else:
                id2pm[_id] = False
    return {k: 1. if v else 0. for k,v in id2pm.items()}

def process_trial_json_entry(line):
    json = j.loads(line)
    sentences = []
    keywords = []
    if 'brief_title' in json:
        sentences.append(pretty(json['brief_title'])) 
    if 'official_title' in json:
        sentences.append(pretty(json['official_title'])) 
    if 'summary' in json:
        sentences.extend(pretty(json['summary']).split(" . ")) 
    if 'description' in json:
        sentences.extend(pretty(json['description']).split(" . ")) 
    if "inclusion" in json:
        sentences.extend(pretty(json["inclusion"]).split(" . ")) 

    if 'primary_purpose' in json:
        keywords.extend(json['primary_purpose'])
    if 'keywords' in json:
        keywords.extend(json["keywords"])
    if 'interventionTypes' in json:
        keywords.extend(set(json['interventionTypes']))
    if 'interventionNames' in json:
        keywords.extend(json['interventionNames']) 
    if 'studyType' in json:
        keywords.extend(json['studyType'])
    if 'conditions' in json: 
        keywords.extend(json['conditions']) 
    keywords = " ".join([x.replace(" ","_") for x in keywords])

    genes = " ".join(json["genes"]) if "genes" in json else " "
    organisms = " ".join(json["organisms"]) if "organisms" in json else " "
    mesh = " ".join(json["meshTags"]) if "meshTags" in json else " "

    _id = json["id"]

    return sentences, mesh, genes, organisms, keywords, _id

def pretty(text):
    return text.lower().replace(". ", " . ").replace(", ", " , ").replace("? ", " . ").replace("! ", " . ")
 
def read_trial_json_file(file_name):
    with open(file_name,"rt", encoding="utf-8", newline="\n") as f:
        for line in f:
            yield process_trial_json_entry(line)       

def read_trial_json(input_data):
    for file_name in glob.glob(input_data.folder+"/"+input_data.json+"/*.json.gz"):
        with gzip.open(file_name,"rt", encoding="utf-8", newline="\n") as f:
            for line in f:
                yield process_trial_json_entry(line)

TrialTokenizers = namedtuple("TrialTokenizers", ["text", "mesh", "gene", "organism", "keywords"])
def read_trial_data(input_data, vocab_size, max_sentences, max_sentence_length, max_bow_size, existing_tokenizers=None):
    id2pm = read_trial_tsv(input_data)
    all_sentences = []
    all_mesh = []
    all_genes = []
    all_organisms = []
    all_keywords = []
    all_id = []
    all_pm = []
    for sentences, mesh, genes, organisms, keywords, _id in read_trial_json(input_data):
        all_sentences.append(sentences)
        all_mesh.append(mesh)
        all_genes.append(genes)
        all_organisms.append(organisms)
        all_keywords.append(keywords)
        all_id.append(_id)
        all_pm.append(id2pm[_id])

    if not vocab_size:
        vocab_size = len({y for x in all_sentences for y in x.split()})
    if not existing_tokenizers:
        t = Tokenizer(num_words=vocab_size)  
        t.fit_on_texts(all_sentences)
    else:
        t = existing_tokenizers.text
    all_sentences = sentence_processing(max_sentences, max_sentence_length, all_sentences, t, split=False)

    num_meshTags = min(len({y for x in all_mesh for y in x.split()}),max_bow_size)
    if not existing_tokenizers:
        mesh_t = Tokenizer(num_words=num_meshTags)
        mesh_t.fit_on_texts(all_mesh)
    else:
        mesh_t = existing_tokenizers.mesh
    all_mesh = mesh_t.texts_to_matrix(all_mesh, mode="binary")

    num_genes =  min(len({y for x in all_genes for y in x.split()}),max_bow_size)
    if not existing_tokenizers:
        gene_t = Tokenizer(num_words=num_genes)
        gene_t.fit_on_texts(all_genes)
    else:
        gene_t = existing_tokenizers.gene
    all_genes = gene_t.texts_to_matrix(all_genes, mode="binary")

    num_orga =  min(len({y for x in all_organisms for y in x.split()}),max_bow_size)
    if not existing_tokenizers:
        orga_t = Tokenizer(num_words=num_orga)
        orga_t.fit_on_texts(all_organisms)
    else:
        orga_t = existing_tokenizers.organism
    all_organisms = orga_t.texts_to_matrix(all_organisms, mode="binary")
   
    num_key =  min(len({y for x in all_keywords for y in x.split()}),max_bow_size)
    if not existing_tokenizers:
        key_t = Tokenizer(num_words=num_key)
        key_t.fit_on_texts(all_keywords)
    else:
        key_t = existing_tokenizers.keywords
    all_keywords = key_t.texts_to_matrix(all_keywords, mode="binary")

    labels = np.array(all_pm)

    if existing_tokenizers:
        return all_sentences, all_mesh, all_genes, all_organisms, all_keywords, all_id, labels, t.word_index
    else:
        return all_sentences, all_mesh, all_genes, all_organisms, all_keywords, all_id, labels, t.word_index, TrialTokenizers(t, mesh_t, gene_t, orga_t, key_t)


def sentence_processing(max_sentences, max_sentence_length, texts, tokenizer, split=True):
    list_of_list_of_sentences = [text.split(" . ") for text in texts] if split else texts
    #trimming number of sentences
    trimmed_list_of_list_of_sentences = []
    for x in list_of_list_of_sentences:
        if len(x) > max_sentences:
            trimmed_list_of_list_of_sentences.append(x[:max_sentences])
        else:
            if len(x) < max_sentences:
                to_pad = max_sentences - len(x)
                for i in range(to_pad):
                    x.append([" "])
            trimmed_list_of_list_of_sentences.append(x)
    list_of_sentence_matrices = [tokenizer.texts_to_sequences(x) for x in trimmed_list_of_list_of_sentences]
    #trimming sentences for length
    trimmed_list_of_sentence_matrices = [keras.preprocessing.sequence.pad_sequences(
                        x, maxlen = max_sentence_length) for x in list_of_sentence_matrices]
    return np.asarray(trimmed_list_of_sentence_matrices)


InputData = namedtuple("InputData", ["tsv","json","other"])
Tokenizers = namedtuple("Tokenizers", ["text", "mesh", "gene", "organism"])
def read_data(input_data,  use_json=True, maxlen=None, vocab_size=None, sentence_wise=None, 
    bonus_info_max=False, return_tokenizers=False, existing_tokenizers=None):
    texts = []
    labels = []
    pmids = []
    mesh_from_tsv = []
    with open(input_data.tsv) as data_file:
        line_number = 0
        for line in data_file:
            if line_number > 0:
                if input_data.other:
                    number, trec_topic_number, trec_doc_id, pm_rel_desc, disease_desc,\
                    gene1_annotation_desc, gene1_name, gene2_annotation_desc,\
                    gene2_name, gene3_annotation_desc, gene3_name, demographics_desc,\
                    other_desc, relevance_score, title, abstract, major_mesh,\
                    minor_mesh, trec_topic_disease, trec_topic_age, trec_topic_sex,\
                    trec_topic_other1, trec_topic_other2, trec_topic_other3 = line.split("\t")
                else:
                    number, trec_topic_number, trec_doc_id, pm_rel_desc, disease_desc,\
                    gene1_annotation_desc, gene1_name, gene2_annotation_desc,\
                    gene2_name, gene3_annotation_desc, gene3_name, demographics_desc,\
                    other_desc, relevance_score, title, abstract, major_mesh,\
                    minor_mesh, trec_topic_disease, trec_topic_age, trec_topic_sex = line.split("\t")
                text = title.replace("[","").replace("]","") + " . " + abstract  #. added
                text = text.lower().replace(". ", " . ").replace(", ", " , ").\
                        replace("? ", " . ").replace("! ", " . ") #jetzt ersetzung mit " . "
                texts.append(text)
                
                mesh_from_tsv.append([x.replace(" ","_") for x in major_mesh.split(";")+minor_mesh.split(";")])

                if pm_rel_desc == 'Human PM' or pm_rel_desc == 'Animal PM':
                    labels.append(1.)
                else:
                    labels.append(0.)
                pmids.append(trec_doc_id)    
            line_number += 1
                   
    #fold_repeats:
    text2indices = {}
    for i, text in enumerate(texts):
        if text in text2indices:
            text2indices[text].append(i)
        else:
            text2indices[text] = [i]
    for text, indices in text2indices.items():
        if len(indices) > 1:
            is_pm = False
            for i in indices:
                if labels[i] == 1.:
                    is_pm = True
            #remove all but first
            for i in indices[1:]:
                labels[i] = None
                texts[i] = None
                pmids[i] = None
                mesh_from_tsv[i] = None
            if is_pm:
                labels[indices[0]] = 1.
    texts = [t for t in texts if t is not None]
    labels = np.array([x for x in labels if x is not None])
    pmids = [x for x in pmids if x is not None]
    mesh_from_tsv = [x for x in mesh_from_tsv if x is not None]
    
    #see https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/
    if not vocab_size:
        vocab_size = len({y for x in texts for y in x.split()})
    if not existing_tokenizers:
        t = Tokenizer(num_words=vocab_size)  
        t.fit_on_texts(texts)
    else:
        t = existing_tokenizers.text
    if sentence_wise:
        texts = sentence_processing(sentence_wise[0], sentence_wise[1], texts, t)
    else:
        texts = t.texts_to_sequences(texts)
        if maxlen:
            texts = keras.preprocessing.sequence.pad_sequences(texts, maxlen=maxlen)
            
    #bonus_info as bow
    max_size = max(int(vocab_size/10), bonus_info_max)
    meshTags = []
    if use_json:
        genes = []
        organisms = []
        pmid2info = parse_json(input_data.json)
        for pmid in pmids:
            info = pmid2info[pmid]
            meshTags.append(info.meshTags)
            genes.append(info.genes)
            organisms.append(info.organisms)

        num_meshTags = min(len({y for x in meshTags for y in x.split()}),max_size)
        if not existing_tokenizers:
            mesh_t = Tokenizer(num_words=num_meshTags)
            mesh_t.fit_on_texts(meshTags)
        else:
            mesh_t = existing_tokenizers.mesh
        meshTags = mesh_t.texts_to_matrix(meshTags, mode="binary")

        num_genes =  min(len({y for x in genes for y in x.split()}),max_size)
        if not existing_tokenizers:
            gene_t = Tokenizer(num_words=num_genes)
            gene_t.fit_on_texts(genes)
        else:
            gene_t = existing_tokenizers.gene
        genes = gene_t.texts_to_matrix(genes, mode="binary")

        num_orga =  min(len({y for x in organisms for y in x.split()}),max_size)
        if not existing_tokenizers:
            orga_t = Tokenizer(num_words=num_orga)
            orga_t.fit_on_texts(organisms)
        else:
            orga_t = existing_tokenizers.organism
        organisms = orga_t.texts_to_matrix(organisms, mode="binary")
        
        if return_tokenizers:
            return texts, labels, t.word_index, meshTags, genes, organisms, Tokenizers(t, mesh_t, gene_t, orga_t)
        else:
            return texts, labels, t.word_index, meshTags, genes, organisms
    else:
        num_meshTags = min(len({y for x in mesh_from_tsv for y in x}),max_size)
        meshTags = [" ".join(x) for x in mesh_from_tsv]
        mesh_t = Tokenizer(num_words=num_meshTags, filters="")
        mesh_t.fit_on_texts(mesh_from_tsv)
        meshTags = mesh_t.texts_to_matrix(mesh_from_tsv, mode="binary")
        if return_tokenizers:
            raise Exception("Not implemented")
        else:
            return texts, labels, t.word_index, meshTags





SmartTuner = namedtuple("SmartTuner",["start_tuning_timestep","timestep2mod"])

def experiment(model_provider, data, labels, nfold=1, stratify=True, max_epochs=9, 
               optimizer="adam", fine_tuner=None, verbose=0):
    if stratify:
        splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    else:
        splitter = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
        
    fold=0
    accs=[]
    for train_index, test_index in splitter.split(np.zeros(len(labels)), labels): #could be used for 10 fold
        if fold >= nfold:
            break
        fold += 1
        #get split
        if not type(data) == list:
            data = [data]
        train_x = []
        test_x = []
        for x in data:
            train_x.append(x[train_index])
            test_x.append(x[test_index])
        train_y = labels[train_index]
        test_y = labels[test_index]
         
        model=model_provider()
        model.compile(loss='binary_crossentropy',
                    optimizer=optimizer,
                    metrics=['acc'])
        
        for i in range(1,max_epochs+1):
            if fine_tuner:
                if i in fine_tuner.timestep2mod:
                    lr = float(keras.backend.get_value(model.optimizer.lr)) * fine_tuner.timestep2mod[i]
                    keras.backend.set_value(model.optimizer.lr, lr)
                    if i == fine_tuner.start_tuning_timestep:
                        model.layers[1].trainable = True
                        model.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['acc'])  
            model.fit(train_x, train_y, epochs=1, batch_size=50, verbose=0)
            re = model.evaluate(train_x, train_y, verbose=0)
            re_acc, re_loss = re[1],re[0]
            te = model.evaluate(test_x, test_y, verbose=0)
            test_acc, test_loss = te[1], te[0]

            line = "Epoch: "+"{:2}".format(fold)+"-"+str(i)+"\tRe-Acc: "+"{:.3f}".format(re_acc)+"\tTest-Acc: "+"{:.3f}".format(test_acc)+"\tRe-Loss: "+"{:.3f}".format(re_loss)+"\tTest-Loss: "+"{:.3f}".format(test_loss)
            if verbose:
                print(line)
        print("Fold:","{:2}".format(fold),"\tRe-Acc:","{:.3f}".format(re_acc),"\tTest-Acc:","{:.3f}".format(test_acc))
        accs.append(test_acc) #no block scope has benefits...
    acc = sum(accs) / nfold
    return acc, accs

def cross_experiment(model_provider, data1, labels1, data2, labels2, max_epochs=9, 
               optimizer="adam", fine_tuner=None, verbose=0):

    if not type(data1) == list:
        data1 = [data1]
    if not type(data2) == list:
        data2 = [data2]
   
    model=model_provider()
    model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['acc'])
    
    for i in range(1,max_epochs+1):
        if fine_tuner:
            if i in fine_tuner.timestep2mod:
                lr = float(keras.backend.get_value(model.optimizer.lr)) * fine_tuner.timestep2mod[i]
                keras.backend.set_value(model.optimizer.lr, lr)
                if i == fine_tuner.start_tuning_timestep:
                    model.layers[1].trainable = True
                    model.compile(loss='binary_crossentropy',
                    optimizer=optimizer,
                    metrics=['acc'])  
        model.fit(data1, labels1, epochs=1, batch_size=50, verbose=0)
        re = model.evaluate(data1, labels1, verbose=0)
        re_acc, re_loss = re[1],re[0]
        te = model.evaluate(data2, labels2, verbose=0)
        test_acc, test_loss = te[1], te[0]

        line = "Epoch: "+str(i)+"\tRe-Acc: "+"{:.3f}".format(re_acc)+"\tTest-Acc: "+"{:.3f}".format(test_acc)+"\tRe-Loss: "+"{:.3f}".format(re_loss)+"\tTest-Loss: "+"{:.3f}".format(test_loss)
        if verbose:
            print(line)
    return test_acc

def train_and_return(model_provider, data, labels, max_epochs=9, 
               optimizer="adam", fine_tuner=None):

    if not type(data) == list:
        data = [data]
   
    model=model_provider()
    model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['acc'])
    
    for i in range(1,max_epochs+1):
        if fine_tuner:
            if i in fine_tuner.timestep2mod:
                lr = float(keras.backend.get_value(model.optimizer.lr)) * fine_tuner.timestep2mod[i]
                keras.backend.set_value(model.optimizer.lr, lr)
                if i == fine_tuner.start_tuning_timestep:
                    model.layers[1].trainable = True
                    model.compile(loss='binary_crossentropy',
                    optimizer=optimizer,
                    metrics=['acc'])  
        model.fit(data, labels, epochs=1, batch_size=50, verbose=0)
        re = model.evaluate(data, labels, verbose=0)
        re_acc, re_loss = re[1],re[0]
        print("Epoch: "+str(i)+"\tRe-Acc: "+"{:.3f}".format(re_acc)+"\tRe-Loss: "+"{:.3f}".format(re_loss))
    return model
