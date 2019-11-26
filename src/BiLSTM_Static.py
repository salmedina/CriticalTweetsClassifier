import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb

class BiLSTM_BERT(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, label_size, use_gpu, batch_size, number_layers, dropout=0.5):
        super(BiLSTM_BERT, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.number_layers = number_layers
        self.label_size = label_size
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
        self.hidden2label = nn.Linear(hidden_dim*2, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        hidden1 = torch.randn(self.number_layers, self.batch_size, self.hidden_dim)
        hidden2 = torch.randn(self.number_layers, self.batch_size, self.hidden_dim)
        if self.use_gpu:
            return hidden1.cuda(), hidden2.cuda()

        return hidden1, hidden2

    def forward(self, encoded_sentence, sentences_length):
        self.hidden = self.init_hidden()
        #x = self.embeddings(sentence).view(self.batch_size, sentence.shape[1], -1)
        #TODO: override x with pretrained embedding. Make sure to bring it to appropriate dimensions to pass through the LSTM
        #x= encoded_sentence
        embed_pack_pad = torch.nn.utils.rnn.pack_padded_sequence(encoded_sentence, sentences_length, batch_first=True)
        ####
        lstm_out, self.hidden = self.lstm(embed_pack_pad, self.hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        X = X.contiguous()
        #Change which state is fed in the fully connected. Now it is the first one, last time was the last one
        y = self.hidden2label(X[:, 0, :])
        ###
        log_probs= F.log_softmax(y, dim=1)
        return log_probs

    def loss(self, y_pred, y, sentences_length):
        y = y.view(-1)
        y_pred = y_pred.view(-1, self.label_size)
        mask = (y > 0).float()
        nb_tokens = int(torch.sum(mask).item())
        y_pred = y_pred[range(y_pred.shape[0]), y] * mask
        ce_loss = -torch.sum(y_pred) / nb_tokens
        # nb_tokens= y.shape[0]
        # loss = nn.CrossEntropyLoss()
        # ce_loss = loss(y_pred, y) / nb_tokens
        return ce_loss


class BiLSTM_BERT_MultiTask(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_layers, event_output_size, crit_output_size, use_gpu, batch_size, dropout=0.5):
        super(BiLSTM_BERT, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.number_layers = num_layers
        self.label_size = output_size
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
        self.hidden2event = nn.Linear(hidden_dim * 2, event_output_size)
        self.hidden2crit = nn.Linear(hidden_dim * 2, crit_output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        hidden1 = torch.randn(self.number_layers, self.batch_size, self.hidden_dim)
        hidden2 = torch.randn(self.number_layers, self.batch_size, self.hidden_dim)
        if self.use_gpu:
            return hidden1.cuda(), hidden2.cuda()

        return hidden1, hidden2

    def forward(self, encoded_sentence, sentences_length):
        self.hidden = self.init_hidden()
        #x = self.embeddings(sentence).view(self.batch_size, sentence.shape[1], -1)
        #TODO: override x with pretrained embedding. Make sure to bring it to appropriate dimensions to pass through the LSTM
        #x= encoded_sentence
        embed_pack_pad = torch.nn.utils.rnn.pack_padded_sequence(encoded_sentence, sentences_length, batch_first=True)
        ####
        lstm_out, self.hidden = self.lstm(embed_pack_pad, self.hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        X = X.contiguous()

        #Change which state is fed in the fully connected. Now it is the first one, last time was the last one
        y_event = self.hidden2event(X[:, 0, :])
        y_crit = self.hidden2crit(X[:, 0, :])

        log_probs_event = F.log_softmax(y_event, dim=1)
        log_probs_crit = F.log_softmax(y_crit, dim=1)
        return log_probs_event, log_probs_crit

    def loss(self, y_pred, y, sentences_length):
        y = y.view(-1)
        y_pred = y_pred.view(-1, self.label_size)
        mask = (y > 0).float()
        nb_tokens = int(torch.sum(mask).item())
        y_pred = y_pred[range(y_pred.shape[0]), y] * mask
        ce_loss = -torch.sum(y_pred) / nb_tokens

        return ce_loss