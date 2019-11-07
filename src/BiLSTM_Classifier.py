import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb


class BiLSTMEventType(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, number_layers, dropout=0.5):
        super(BiLSTMEventType, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.number_layers= number_layers
        self.label_size= label_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first= True)
        self.hidden2label = nn.Linear(hidden_dim*2, label_size)
        self.hidden = self.init_hidden()


        # #self.embedding_dim = embedding_dim
        # self.hidden_dim = hidden_dim
        # self.vocab_size = vocab_size
        # #self.labels = labels
        # self.labels_size = label_size
        # self.batch_Size = batch_size
        # self.number_layers = number_layers
        # self.word_embeds = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        #
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
        #                     num_layers=number_layers, bidirectional=True, batch_first=True)
        # self.hidden2tag = nn.Linear(hidden_dim, self.labels_size)
        # self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        hidden1 = torch.randn(self.number_layers, self.batch_size, self.hidden_dim)
        hidden2 = torch.randn(self.number_layers, self.batch_size, self.hidden_dim)
        if self.use_gpu:
            return (hidden1.cuda(), hidden2.cuda())
        return (hidden1, hidden2)
        # if self.use_gpu:
        #     return torch.zeros(2, self.batch_size, self.hidden_dim).cuda()), torch.zeros(2, self.batch_size, self.hidden_dim).cuda())
        # else:
        #     return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
        #             Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))



    def forward(self, sentence):
        self.hidden = self.init_hidden()
        x = self.embeddings(sentence).view(self.batch_size, sentence.shape[1], -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2label(lstm_out[:, -1, :])
        log_probs = F.log_softmax(y, dim=1)
        return log_probs


    def loss(self, y_pred, y, sentences_length):
        #loss = nn.NLLLoss()
        y = y.view(-1)
        y_pred= y_pred.view(-1, self.label_size)
        mask = (y > 0).float()
        nb_tokens = int(torch.sum(mask).data[0])
        #nb_tokens = int(torch.sum(mask).item(0))
        y_pred = y_pred[range(y_pred.shape[0]), y] * mask
        ce_loss = -torch.sum(y_pred) / nb_tokens
        return  ce_loss