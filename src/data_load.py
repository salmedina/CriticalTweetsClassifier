from bert_embedding import BertEmbedding
import torch
import random
import json
import pdb
import math
import numpy as np

bert = BertEmbedding(max_seq_length=50)

def saveBERT(embedding_file, data_file='../data/labeled_data.json'):
    '''
    This function is required to run if there is a new dataset
    '''
    f = open(data_file)
    data = json.load(f)
    sentences = list()
    ids = list()

    for id in data:
        text = data[id]['text']
        if len(text) > 0:
            sentences.append(data[id]['text'])
            ids.append(id)

    print("Data loaded...")
    #bert_encoding = bert(sentences)
    print("Data processed...")

    # Encode the sentences in batches of size batch_sz
    batch_sz = 10
    bert_embeddings = dict()
    for i in range(0, len(ids), batch_sz):
        bert_encoding = bert(sentences[i:i+batch_sz])
        print(f'{i}/{len(ids)}')
        for j in range(batch_sz):
            id = ids[i+j]
            bert_embed = bert_encoding[j][1]
            #TODO: Check if it is necessary to change from torch to list then torch during training
            x_tensor = torch.tensor(bert_embed, dtype=torch.float)
            vector = x_tensor.tolist()
            bert_embeddings[id] = vector

    np.save(embedding_file, bert_embeddings)


def encodeSentence(sentence, id, embeddings, vocab, embedding_type):
    x_vector=[]
    words= sentence.split(' ')

    #Data already pre-processed and clean!
    if embedding_type == 'bert':
        if id in embeddings:
            bert_vector = embeddings[id]
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


def loadData(embedding_type, event_type, data_path, data_type='labeled'):
    f = open(data_path)

    vocab = {'<PAD>': 0}
    data = json.load(f)
    X = list()
    Y_cr = list()
    Y_event = list()
    events = {}
    ids = []
    for id in data:
        event = data[id]['event'].lower()
        if event_type in event:
            if event not in events:
                events[event] = len(events)
            ids.append(id)
            X.append(data[id]['text'])
            Y_event.append(events[event])
            if data_type == 'labeled':
                Y_cr.append(0 if data[id]['label'] == 'low' else 1)

    indices = [i for i in range(len(X))]
    random.shuffle(indices)
    split = math.ceil(len(X)*0.7)

    print("Loading embeddings...")
    embeddings = loadEmbeddings(set(ids), embedding_type=embedding_type, embedding_file='../data/bert_embeddings.npy')
    print("Embeddings loaded...")

    train = list()
    val = list()
    for i in indices:
        if len(X[i]) > 0:
            x_i = encodeSentence(X[i], ids[i], embeddings, vocab, embedding_type)
            y_i = torch.tensor(Y_event[i], dtype=torch.long)
            if i < split:
                train.append((x_i, y_i, Y_cr[i]))
            else:
                val.append((x_i, y_i, Y_cr[i]))

    return train, val, events, vocab


def loadEmbeddings(ids, embedding_type='torch', embedding_file= None):
    embeddings = {}
    if embedding_type == 'glove':
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
