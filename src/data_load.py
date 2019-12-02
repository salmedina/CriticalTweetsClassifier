from bert_embedding import BertEmbedding
import torch
import random
import json
import yaml
import math
import numpy as np
from easydict import EasyDict as edict

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


def encodeTweet(sentence, id, embeddings, vocab, embedding_type):
    x_vector = []
    words = sentence.split(' ')

    if embedding_type == 'bert':
        # BERT embeddings were already pre-procssed and linked through id
        if id in embeddings:
            bert_vector = embeddings[id]
            x_tensor = torch.tensor(bert_vector)
            return x_tensor
        bert_encode = bert([sentence])
        bert_embed = bert_encode[0][1]
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
            vocab[word] = [len(vocab)]
        word_embed = vocab[word]
        x_vector.append(word_embed)

    x_tensor = torch.tensor(x_vector, dtype=torch.long)

    return x_tensor


def loadEmbeddings(embedding_type='torch', embeddings_path=None):
    embeddings = {}
    if embedding_type == 'glove':
        f = open(embeddings_path)
        lines = f.read().split('\n')[:-1]
        f.close()
        for line in lines:
            vector = line.split(' ')
            word = vector[0]
            vector = [float(i) for i in vector[1:]]
            embeddings[word] = vector
        embeddings['UNK'] = len(vector)*[0.0]
    elif embedding_type == 'bert':
        embeddings = np.load(embeddings_path).item()

    return embeddings


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
        include = False
        for e in event_type.split(','):
            if e in event:
                include = True
        if include:
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
    embeddings_path = dict(glove='../data/glove.6B.100d.txt',
                           bert='../data/bert_embeddings.npy')
    embeddings = loadEmbeddings(embedding_type=embedding_type,
                                embeddings_path=embeddings_path[embedding_type])

    train = list()
    val = list()
    for i in indices:
        if len(X[i]) > 0:
            x_i = encodeTweet(X[i], ids[i], embeddings, vocab, embedding_type)
            y_i = torch.tensor(Y_event[i], dtype=torch.long)
            if i < split:
                train.append((x_i, y_i, Y_cr[i]))
            else:
                val.append((x_i, y_i, Y_cr[i]))

    return train, val, events, vocab


def loadExperimentData(desc_path, embedding_type, data_path, data_type='labeled'):
    # Load data split from experiment descriptor file
    experiment_split = edict(yaml.load(desc_path, Loader=yaml.FullLoader))
    events_idx = {event: idx for idx, event in enumerate(experiment_split.train)}

    # Load data from json file
    f = open(data_path)
    data = json.load(f)

    # TODO: add embeddings file to the script argument, cannot be hard coded at this level
    print("Loading embeddings...")
    embeddings_path = dict(glove='../data/glove.6B.100d.txt',
                           bert='../data/bert_embeddings.npy')
    embeddings = loadEmbeddings(embedding_type=embedding_type,
                                embeddings_path=embeddings_path[embedding_type])

    # Load data into structures
    vocab = {'<PAD>': 0}
    train = list()
    val = list()
    for id in data:
        event = data[id]['event'].lower()

        x = encodeTweet(data[id]['text'], id, embeddings, vocab, embedding_type)
        y_event = torch.tensor(events_idx[event], dtype=torch.long)
        y_cr = 0 if data[id]['label'] == 'low' else 1

        if event in experiment_split.train:
            train.append((x, y_event, y_cr))
        elif event in experiment_split.valid:
            val.append((x, y_event, y_cr))

    # Shuffle the samples and split them into train and val
    random.shuffle(train)
    random.shuffle(val)

    return train, val, events_idx, vocab
