import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMEventType(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, num_layers, pretrained_embeds= None,
                 frozen= False, dropout=0.5):
        super(BiLSTMEventType, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.number_layers= num_layers
        self.label_size= label_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        ####
        if pretrained_embeds!= None:
            self.embeddings.load_state_dict({'weight': pretrained_embeds})
            if frozen:
                self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first= True,
                            num_layers= self.number_layers, dropout= self.dropout)

        self.hidden2label = nn.Linear(hidden_dim*2, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        hidden1 = torch.randn(2*self.number_layers, self.batch_size, self.hidden_dim)
        hidden2 = torch.randn(2*self.number_layers, self.batch_size, self.hidden_dim)
        if self.use_gpu:
            return (hidden1.cuda(), hidden2.cuda())
        return (hidden1, hidden2)


    def forward(self, sentence, sentences_length):
        self.hidden = self.init_hidden()
        x = self.embeddings(sentence).view(self.batch_size, sentence.shape[1], -1)
        embed_pack_pad = torch.nn.utils.rnn.pack_padded_sequence(x, sentences_length, batch_first=True)
        lstm_out, self.hidden = self.lstm(embed_pack_pad, self.hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        X = X.contiguous()

        idx1 = torch.tensor([i - 1 for i in sentences_length])
        X = X.view(self.batch_size, sentences_length[0], 2, self.hidden_dim)
        # Forward LSTM
        x1 = X[torch.arange(X.shape[0]), idx1][:, 0, :]
        # Backward LSTM
        x2 = X[:, 0, 1, :]
        X = torch.cat((x1, x2), dim=1)

        y = self.hidden2label(X)

        # probs = F.softmax(y, dim=1)
        log_probs = F.log_softmax(y, dim=1)
        return log_probs

    def loss(self, y_pred, y, sentences_length):
        y = y.view(-1)
        y_pred= y_pred.view(-1, self.label_size)
        mask = (y > 0).float()
        nb_tokens = int(torch.sum(mask).item())
        y_pred = y_pred[range(y_pred.shape[0]), y] * mask
        ce_loss = -torch.sum(y_pred) / nb_tokens
        # nb_tokens= y.shape[0]
        # loss = nn.CrossEntropyLoss()
        # ce_loss = loss(y_pred, y) / nb_tokens
        return  ce_loss




