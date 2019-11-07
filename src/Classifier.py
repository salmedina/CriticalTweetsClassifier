import json
import random
import math
import torch
import torch.optim as optim
from BiLSTM_Classifier import BiLSTMEventType
import torch.nn.functional as F
import pdb

random.seed(1107)
torch.manual_seed(1107)

def setEmbeddings():
    file = '../data/glove.6B.100d.txt'
    embeddings = {}
    f = open(file)
    lines = f.read().split('\n')[:-1]
    f.close()
    for line in lines:
        vector = line.split(' ')
        word = vector[0]
        vector = [float(i) for i in vector[1:]]
        embeddings[word] = vector
    embeddings['UNK'] = len(vector) * [0.0]
    return embeddings


def loadData(embeddings, event_type='earthquake', data_type= 'labeled'):
    if data_type== 'labeled': f=open('../data/labeled_data.json')
    else: f=open('../data/unlabeled_data.json')
    vocab={'<PAD>':0}
    data= json.load(f)
    X= []
    Y_cr=[]
    Y_event=[]
    events= {'<PAD>':0}
    for id in data:
        event = data[id]['event'].lower()
        if event_type in event:
            if event not in events:
                events[event]= len(events)
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
    for i in indices:
        x_i= encodeSentence(X[i], embeddings, vocab)
        y_i= torch.tensor(Y_event[i], dtype=torch.long)
        if i<split:
            train.append((x_i, y_i, Y_cr[i]))
        else:
            val.append((x_i, y_i, Y_cr[i]))
    # train_X= [encodeSentence(X[i], embeddings, vocab) for i in indices[:split]]
    # #train_Y_event= torch.tensor([Y_event[i] for i in indices[:split]], dtype=torch.long)
    # train_Y_event = [torch.tensor(Y_event[i], dtype=torch.long) for i in indices[:split]]
    # train_Y_cr = [Y_cr[i] for i in indices[:split]]
    #
    # val_X = [encodeSentence(X[i], embeddings, vocab) for i in indices[split:]]
    # #val_Y_event = torch.tensor([Y_event[i] for i in indices[split:]], dtype=torch.long)
    # val_Y_event = [torch.tensor(Y_event[i], dtype=torch.long) for i in indices[split:]]
    # val_Y_cr = [Y_cr[i] for i in indices[split:]]
    #
    # train= (train_X, train_Y_cr, train_Y_event)
    # val= (val_X, val_Y_cr, val_Y_event)

    return train, val, events, vocab


#TODO: Figure out the BERT Encoding; currently using torch embeddings, use BERT frozen instead. Check resutls for both
def encodeSentence(sentence, embeddings, vocab, use_embeddings=False):
    x_vector=[]
    words= sentence.split(' ')
    #Data already preprocessec and clean!
    for word in words:
        word = word.lower()
        if use_embeddings:
            word_embedding = embeddings['UNK']
            if word in embeddings:
                word_embedding = embeddings[word]
        else:
            if word not in vocab:
                vocab[word]= [len(vocab)]
            word_embedding = vocab[word]
        x_vector.append(word_embedding)
    x_tensor = torch.tensor(x_vector, dtype=torch.long)
    return x_tensor

def maxCriterion(element):
    return len(element[0])


def batchify(data, batch_size, randomize= True):
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
        x_tensor= torch.zeros((batch_size, dim, 1), dtype=torch.long)
        y_tensor= torch.zeros((batch_size), dtype=torch.long)
        for i in range(batch_size):
            x_i, y_i, y_cr= batch[i]
            x_i= F.pad(x_i, (0, 0, 0, dim-x_i.shape[0]))
            #y_i = F.pad(y_i, (0, dim - y_i.shape[0]))
            x_tensor[i]= x_i
            y_tensor[i] = y_i
        batches.append((x_tensor, y_tensor, real_lengths))
    return batches

#TODO: Sanity-check/ does cuda work? Test it on some GPU
def train_model(batch_size, embedding_dim, hidden_dim, number_layers=2, epochs=5, use_gpu=False):

    #embeddings= setEmbeddings()
    embeddings={}
    print('Training model')
    train, val, events, vocab= loadData(embeddings)
    #train= batchify(train, batch_size)
    val = batchify(val, batch_size, randomize=False)

    #loss_ce = F.cross_entropy
    model = BiLSTMEventType(embedding_dim, hidden_dim, len(vocab), len(events), use_gpu, batch_size, number_layers)
    if use_gpu:
        model= model.cuda()
    #optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.SGD(model.parameters(), lr=0.03, weight_decay=1e-4)
    # with torch.no_grad():
    #     train_i = batchify(train, batch_size)
    #model.train()

    best_f1= 0.0
    for epoch in range(epochs):
        print('Epoch Number:')
        print(epoch)
        total_Loss = 0.
        train_i = batchify(train, batch_size)
        for x, y, seq_lengths in train_i:
            model.zero_grad()
            y_pred = model(x)
            #myLoss = loss_ce(y_pred, y, seq_lengths)
            myLoss = model.loss(y_pred, y, seq_lengths)
            total_Loss += myLoss.data[0]
            #Some issue here: why retaining graph???
            #myLoss.backward(retain_graph=True)
            myLoss.backward()
            optimizer.step()
            del myLoss
        with torch.no_grad():
            print("Loss: Train set", total_Loss)
            # self.test(model, train_i)
            print("Evaluation: Dev set")
            f1 = test(model, val)
            if f1 < best_f1:
                print('Early Convergence!!!!')
                print(best_f1)
                break
            else:
                best_f1 = f1
    return model

#TODO; implement the test for convergence criteria to occur/ used for validation/testing
def test(model, data):
    return 0

train_model(16, 300, 100)