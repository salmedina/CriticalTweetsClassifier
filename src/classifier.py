import json
import random
import math
import torch
import torch.optim as optim
from BiLSTM_Classifier import BiLSTMEventType
from BiLSTM_Static import BiLSTM_BERT
import torch.nn.functional as F
from bert_embedding import BertEmbedding
from data_load import *
import pdb

random.seed(1107)
torch.manual_seed(1)


def maxCriterion(element):
    return len(element[0])


def batchify(data, batch_size, embedding_dim=1, randomize= True):
    data= data
    batches=[]
    num_batches= len(data) //batch_size
    leftover= len(data) - num_batches*batch_size
    if randomize:
        random.shuffle(data)
    else:
        data=data+ data[:batch_size-leftover]
        num_batches+=1
    for b in range(num_batches):
        # print('Batchifying')
        # pdb.set_trace()
        batch= data[b*batch_size:(b+1)*batch_size]
        batch= sorted(batch, key= maxCriterion, reverse=True)
        dim= batch[0][0].shape[0]
        real_lengths = [i[0].shape[0] for i in batch]
        if embedding_dim==1:
            x_tensor= torch.zeros((batch_size, dim, embedding_dim), dtype=torch.long)
        else:
            x_tensor = torch.zeros((batch_size, dim, embedding_dim), dtype=torch.float)
        y_tensor= torch.zeros((batch_size), dtype=torch.long)
        for i in range(batch_size):
            x_i, y_i, y_cr= batch[i]
            #pdb.set_trace()
            x_i= F.pad(x_i, (0, 0, 0, dim-x_i.shape[0]))
            #y_i = F.pad(y_i, (0, dim - y_i.shape[0]))
            x_tensor[i]= x_i
            y_tensor[i] = y_i
        batches.append((x_tensor, y_tensor, real_lengths))
    return batches

#TODO: Sanity-check/ does cuda work? Test it on some GPU
def train_model(batch_size, embedding_dim, hidden_dim, embedding_type, event_type, number_layers=2, epochs=5, use_gpu=False):
    #embeddings={}
    print("Loading Data....")
    train, val, events, vocab= load_data(embedding_type, event_type= event_type)
    print('Training model...')
    if embedding_type== 'bert' or embedding_type== 'glove':
        embedding_dim= train[0][0].shape[1]
        val = batchify(val, batch_size, embedding_dim = embedding_dim, randomize=False)
        model= BiLSTM_BERT(embedding_dim, hidden_dim, len(events), use_gpu, batch_size, number_layers)
    else:
        val = batchify(val, batch_size, randomize=False)
        model = BiLSTMEventType(embedding_dim, hidden_dim, len(vocab), len(events), use_gpu, batch_size, number_layers)
    if use_gpu:
        model= model.cuda()
    #optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.SGD(model.parameters(), lr=0.03, weight_decay=1e-4)
    #model.train()
    best_f1= 0.0
    for epoch in range(epochs):
        print('Epoch Number:')
        print(epoch)
        total_Loss = 0.
        if embedding_type== 'bert' or embedding_type== 'glove':
            train_i = batchify(train, batch_size, embedding_dim= embedding_dim)
        else:
            train_i = batchify(train, batch_size)
        for x, y, seq_lengths in train_i:
            model.zero_grad()
            y_pred = model(x, seq_lengths)
            #myLoss = loss_ce(y_pred, y, seq_lengths)
            myLoss = model.loss(y_pred, y, seq_lengths)
            total_Loss += myLoss.item()
            myLoss.backward()
            optimizer.step()
            del myLoss
        with torch.no_grad():
            print("Loss: Train set ", total_Loss)
            accuracy, f1, final_metrics = test(model, train_i, events)
            print("Accuracy & F1: Train set ", accuracy, f1)
            print(final_metrics)
            # self.test(model, train_i)
            accuracy, f1, final_metrics = test(model, val, events)
            print("Accuracy & F1 on Dev set ", accuracy, f1)
            print(final_metrics)
            if f1 < best_f1:
                print('Early Convergence!!!!')
                print(best_f1)
                break
            else:
                best_f1 = f1
    return model


def test(model, data, events):
    correct=0.0
    total=0.0
    event_scores= {event: {'correct':0.0, 'gold':0.0001, 'predicted':0.0001} for event in range(len(events))}
    for x, y, seq_lengths in data:
        total+= len(y)
        y_pred = model(x, seq_lengths)
        y_pred_value= torch.argmax(y_pred, 1)
        vector=  y-y_pred_value
        correct+= (vector==0).sum().item()
        for i in range(len(y)):
            actual= y[i].item()
            pred= y_pred_value[i].item()
            event_scores[actual]['gold'] += 1
            event_scores[pred]['predicted'] += 1
            if actual==pred:
                event_scores[actual]['correct']+=1
    final_metrics= {}
    macro_f1=0.0
    for event_id in event_scores:
        for e, v in events.items():
            if v== event_id:
                event= e
        pr= event_scores[event_id]['correct']/event_scores[event_id]['predicted']
        re = event_scores[event_id]['correct'] / event_scores[event_id]['gold']
        f1= (2*pr*re)/(pr+re+0.0001)
        if event_id!=0:
            final_metrics[event]= (pr, re, f1)
            macro_f1+= f1
    accuracy= correct/total
    # print('event Scores')
    # print(event_scores)
    macro_f1/=len(final_metrics)
    return accuracy, macro_f1, final_metrics

train_model(16, 300, 100, 'bert', 'earthquake')