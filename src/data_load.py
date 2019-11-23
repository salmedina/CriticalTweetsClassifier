from bert_embedding import BertEmbedding
import torch
import random
import json
import pdb
import math
import numpy as np

bert = BertEmbedding()

#Need to run it if in a new dataset
def saveBERT(embedding_file, data_file='../data/labeled_data.json'):
    f = open(data_file)
    data = json.load(f)
    sentences=[]
    ids=[]
    for id in data:
        sentences.append(data[id]['text'])
        ids.append(id)
    print("Data loaded...")
    bert_encoding = bert(sentences)
    print("Data processed...")
    bert_embeddings={}
    #f = open(embedding_file, 'w')
    for i in range(len(ids)):
        id= ids[i]
        bert_embed= bert_encoding[i][1]
        x_tensor = torch.tensor(bert_embed, dtype=torch.float)
        vector = x_tensor.tolist()
        #f.write(id+'\t'+vector+'\n')
        bert_embeddings[id]= vector
    f= open(embedding_file, 'w')
    json.dump(bert_embeddings, f)
    f.close()

def encodeSentence(sentence, id, embeddings, vocab, embedding_type):
    x_vector=[]
    words= sentence.split(' ')
    #Data already preprocessec and clean!
    if embedding_type == 'bert':
        if id in embeddings:
            bert_vector= embeddings[id]
            x_tensor = torch.tensor(bert_vector)
            return x_tensor
        bert_encode= bert([sentence])
        bert_embed= bert_encode[0][1]
        x_tensor = torch.tensor(bert_embed, dtype=torch.float)
        return x_tensor
    elif embedding_type == 'glove':
        for word in words:
            word = word.lower()
            glove_embed = embeddings['UNK']
            if word in embeddings:
                glove_embed = embeddings[word]
            x_vector.append(glove_embed)
        x_tensor = torch.tensor(x_vector, dtype=torch.float)
        return x_tensor
    for word in words:
        word = word.lower()
        if word not in vocab:
            vocab[word]= [len(vocab)]
        word_embed = vocab[word]
        x_vector.append(word_embed)
    x_tensor = torch.tensor(x_vector, dtype=torch.long)
    return x_tensor

def load_data(embedding_type, event_type='', data_type= 'labeled'):
    if data_type== 'labeled': f=open('../data/labeled_data.json')
    else: f=open('../data/unlabeled_data.json')
    vocab={'<PAD>':0}
    data= json.load(f)
    X= []
    Y_cr=[]
    Y_event=[]
    events= {'<PAD>':0}
    ids=[]
    for id in data:
        event = data[id]['event'].lower()
        if event_type in event:
            if event not in events:
                events[event]= len(events)
            ids.append(id)
            X.append(data[id]['text'])
            Y_event.append(events[event])
            if data_type == 'labeled':
                if data[id]['label']== 'low':
                    Y_cr.append(0)
                else:
                    Y_cr.append(1)
    indices= [i for i in range(len(X))]
    random.shuffle(indices)
    split= math.ceil(len(X)*0.7)
    train=[]
    val=[]
    print("Loading embeddings...")
    embeddings = loadEmbeddings(set(ids), embedding_type=embedding_type, embedding_file= '../data/bert_embeddings.npy')

    print("Embeddings loaded...")
    for i in indices:
        if len(X[i])>0:
            x_i= encodeSentence(X[i], ids[i], embeddings, vocab, embedding_type)
            y_i= torch.tensor(Y_event[i], dtype=torch.long)
            if i<split:
                train.append((x_i, y_i, Y_cr[i]))
            else:
                val.append((x_i, y_i, Y_cr[i]))
    return train, val, events, vocab


def loadEmbeddings(ids, embedding_type= 'torch', embedding_file= None):
    embeddings = {}
    if embedding_type=='glove':
        file = '../data/glove.6B.100d.txt'
        f = open(file)
        lines = f.read().split('\n')[:-1]
        f.close()
        for line in lines:
            vector = line.split(' ')
            word = vector[0]
            vector = [float(i) for i in vector[1:]]
            #embeddings[word] = torch.tensor(vector)
            embeddings[word] = vector
        #embeddings['UNK'] = torch.zeros((len(vector)))
        embeddings['UNK'] = len(vector)*[0.0]
    elif embedding_type== 'bert':
        embeddings = np.load(embedding_file).item()

    return embeddings

#saveBERT('../data/bert_embeddings.json')
