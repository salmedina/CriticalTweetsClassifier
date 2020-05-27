import torch
import torch.nn as nn
import torch.nn.functional as F
from revgrad import RevGrad


class BiLSTM_BERT(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, label_size, use_gpu, batch_size, num_layers, dropout=0.5):
        super(BiLSTM_BERT, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.number_layers = num_layers
        self.label_size = label_size
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True,
                            num_layers= self.number_layers, dropout= self.dropout)
        self.hidden2label = nn.Linear(hidden_dim*2, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        hidden1 = torch.randn(2*self.number_layers, self.batch_size, self.hidden_dim)
        hidden2 = torch.randn(2*self.number_layers, self.batch_size, self.hidden_dim)
        if self.use_gpu:
            return hidden1.cuda(), hidden2.cuda()

        return hidden1, hidden2

    def forward(self, encoded_sentence, sentences_length):
        self.hidden = self.init_hidden()
        embed_pack_pad = torch.nn.utils.rnn.pack_padded_sequence(encoded_sentence, sentences_length, batch_first=True)
        lstm_out, self.hidden = self.lstm(embed_pack_pad, self.hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        X = X.contiguous()

        idx1 = torch.tensor([i - 1 for i in sentences_length])
        X = X.view(self.batch_size, sentences_length[0], 2, self.hidden_dim)
        #Forward LSTM
        x1 = X[torch.arange(X.shape[0]), idx1][:, 0, :]
        #Backward LSTM
        x2 = X[:, 0, 1, :]
        X= torch.cat((x1, x2), dim=1)

        y = self.hidden2label(X)

        log_probs = F.log_softmax(y, dim=1)
        return log_probs, X

    def loss(self, y_pred, y, sentences_length):
        y = y.view(-1)
        y_pred = y_pred.view(-1, self.label_size)
        mask = (y >= 0).float()
        nb_tokens = int(torch.sum(mask).item())
        y_pred = y_pred[range(y_pred.shape[0]), y] * mask
        ce_loss = -torch.sum(y_pred) / nb_tokens

        return ce_loss


class BiLSTM_BERT_Attention(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, label_size, use_gpu, batch_size, number_layers, dropout=0.5):
        super(BiLSTM_BERT, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.number_layers = number_layers
        self.label_size = label_size
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True,
                            num_layers= self.number_layers, dropout= self.dropout)
        self.hidden2label = nn.Linear(hidden_dim*2, label_size)
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.query_vector = nn.Parameter(torch.Tensor(hidden_dim * 2))
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        hidden1 = torch.randn(2*self.number_layers, self.batch_size, self.hidden_dim)
        hidden2 = torch.randn(2*self.number_layers, self.batch_size, self.hidden_dim)
        if self.use_gpu:
            return hidden1.cuda(), hidden2.cuda()

        return hidden1, hidden2

    def forward(self, encoded_sentence, sentences_length):
        self.hidden = self.init_hidden()
        embed_pack_pad = torch.nn.utils.rnn.pack_padded_sequence(encoded_sentence, sentences_length, batch_first=True)
        lstm_out, self.hidden = self.lstm(embed_pack_pad, self.hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        X = X.contiguous()
        scores = torch.mv(self.attn(X).view(-1, self.hidden_dim * 2), self.query_vector).view(self.batch_size, self.seq_len, 1)
        Xsummary = torch.bmm(X, scores).view(self.batch_size, self.hidden_dim * 2)  # feed it to the clf

        #Modified version, check if this works....
        idx1 = torch.tensor([i - 1 for i in sentences_length])
        Xsummary = Xsummary.view(self.batch_size, sentences_length[0], 2, self.hidden_dim)
        #Forward LSTM
        x1 = Xsummary[torch.arange(Xsummary.shape[0]), idx1][:, 0, :]
        #Backward LSTM
        x2 = Xsummary[:, 0, 1, :]
        Xsummary= torch.cat((x1, x2), dim=1)

        y = self.hidden2label(Xsummary[:, 0, :])
        log_probs= F.log_softmax(y, dim=1)
        return log_probs

    def loss(self, y_pred, y, sentences_length):
        y = y.view(-1)
        y_pred = y_pred.view(-1, self.label_size)
        mask = (y >= 0).float()
        nb_tokens = int(torch.sum(mask).item())
        y_pred = y_pred[range(y_pred.shape[0]), y] * mask
        ce_loss = -torch.sum(y_pred) / nb_tokens

        return ce_loss


class BiLSTM_BERT_MultiTask(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_layers, event_output_size, crit_output_size, use_gpu, batch_size, dropout=0.5):
        super(BiLSTM_BERT_MultiTask, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.number_layers = num_layers
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True,
                            num_layers=self.number_layers, dropout= self.dropout)
        self.hidden2event = nn.Linear(hidden_dim * 2, event_output_size)
        self.hidden2crit = nn.Linear(hidden_dim * 2, crit_output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        hidden1 = torch.randn(2*self.number_layers, self.batch_size, self.hidden_dim)
        hidden2 = torch.randn(2*self.number_layers, self.batch_size, self.hidden_dim)
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
        embeddings = X[:, 0, :]

        idx1 = torch.tensor([i - 1 for i in sentences_length])
        X = X.view(self.batch_size, sentences_length[0], 2, self.hidden_dim)
        #Forward LSTM
        x1 = X[torch.arange(X.shape[0]), idx1][:, 0, :]
        #Backward LSTM
        x2 = X[:, 0, 1, :]
        X= torch.cat((x1, x2), dim=1)


        #Change which state is fed in the fully connected. Now it is the first one, last time was the last one
        y_event = self.hidden2event(embeddings)
        y_crit = self.hidden2crit(embeddings)

        log_probs_event = F.log_softmax(y_event, dim=1)
        log_probs_crit = F.log_softmax(y_crit, dim=1)

        return log_probs_event, log_probs_crit, embeddings


    def loss(self, y_pred, y, sentences_length, output_size):
        y = y.view(-1)
        y_pred = y_pred.view(-1, output_size)
        mask = (y >= 0).float()
        nb_tokens = int(torch.sum(mask).item())
        y_pred = y_pred[range(y_pred.shape[0]), y] * mask
        ce_loss = -torch.sum(y_pred) / nb_tokens

        return ce_loss

class BiLSTM_BERT_Adversarial(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_layers, event_output_size, crit_output_size, use_gpu, batch_size, adv_scale=100.0, dropout=0.5):
        super(BiLSTM_BERT_Adversarial, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.number_layers = num_layers
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True,
                            num_layers= self.number_layers, dropout= self.dropout)
        self.hidden2crit = nn.Linear(hidden_dim * 2, crit_output_size)
        self.hidden2event = nn.Sequential(RevGrad(scale=adv_scale),
                                          nn.Linear(hidden_dim * 2, event_output_size))
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        hidden1 = torch.randn(2*self.number_layers, self.batch_size, self.hidden_dim)
        hidden2 = torch.randn(2*self.number_layers, self.batch_size, self.hidden_dim)
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
        embeddings = X[:, 0, :]

        idx1 = torch.tensor([i - 1 for i in sentences_length])
        X = X.view(self.batch_size, sentences_length[0], 2, self.hidden_dim)
        #Forward LSTM
        x1 = X[torch.arange(X.shape[0]), idx1][:, 0, :]
        #Backward LSTM
        x2 = X[:, 0, 1, :]
        X= torch.cat((x1, x2), dim=1)

        #Change which state is fed in the fully connected. Now it is the first one, last time was the last one
        y_event = self.hidden2event(embeddings)
        y_crit = self.hidden2crit(embeddings)

        log_probs_event = F.log_softmax(y_event, dim=1)
        log_probs_crit = F.log_softmax(y_crit, dim=1)

        return log_probs_event, log_probs_crit, embeddings

    def loss(self, y_pred, y, sentences_length, output_size):
        y = y.view(-1)
        y_pred = y_pred.view(-1, output_size)
        mask = (y >= 0).float()
        nb_tokens = int(torch.sum(mask).item())
        y_pred = y_pred[range(y_pred.shape[0]), y] * mask
        ce_loss = -torch.sum(y_pred) / nb_tokens

        return ce_loss


class BiLSTM_BERT_Adversarial_Attention(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_layers, event_output_size, crit_output_size, use_gpu, batch_size, dropout=0.5):
        super(BiLSTM_BERT_Adversarial, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.number_layers = num_layers
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True,
                            num_layers= self.number_layers, dropout= self.dropout)
        self.hidden2crit = nn.Linear(hidden_dim * 2, crit_output_size)
        self.hidden2event = nn.Sequential(RevGrad(),
                                          nn.Linear(hidden_dim * 2, event_output_size))
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        hidden1 = torch.randn(2*self.number_layers, self.batch_size, self.hidden_dim)
        hidden2 = torch.randn(2*self.number_layers, self.batch_size, self.hidden_dim)
        if self.use_gpu:
            return hidden1.cuda(), hidden2.cuda()

        return hidden1, hidden2

    def forward(self, encoded_sentence, sentences_length):
        self.hidden = self.init_hidden()
        embed_pack_pad = torch.nn.utils.rnn.pack_padded_sequence(encoded_sentence, sentences_length, batch_first=True)
        lstm_out, self.hidden = self.lstm(embed_pack_pad, self.hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        X = X.contiguous()

        idx1 = torch.tensor([i - 1 for i in sentences_length])
        X = X.view(self.batch_size, sentences_length[0], 2, self.hidden_dim)
        #Forward LSTM
        x1 = X[torch.arange(X.shape[0]), idx1][:, 0, :]
        #Backward LSTM
        x2 = X[:, 0, 1, :]
        X= torch.cat((x1, x2), dim=1)

        #Change which state is fed in the fully connected. Now it is the first one, last time was the last one
        y_event = self.hidden2event(X[:, 0, :])
        y_crit = self.hidden2crit(X[:, 0, :])

        log_probs_event = F.log_softmax(y_event, dim=1)
        log_probs_crit = F.log_softmax(y_crit, dim=1)

        return log_probs_event, log_probs_crit

    def loss(self, y_pred, y, sentences_length, output_size):
        y = y.view(-1)
        y_pred = y_pred.view(-1, output_size)
        mask = (y >= 0).float()
        nb_tokens = int(torch.sum(mask).item())
        y_pred = y_pred[range(y_pred.shape[0]), y] * mask
        ce_loss = -torch.sum(y_pred) / nb_tokens

        return ce_loss