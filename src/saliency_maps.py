import random
import torch
import torch.optim as optim
import numpy as np
from BiLSTM_Classifier import BiLSTMEventType
from BiLSTM_Static import BiLSTM_BERT, BiLSTM_BERT_MultiTask, BiLSTM_BERT_Adversarial
from classifier import batchify, maxCriterion
import torch.nn.functional as F
from bert_embedding import BertEmbedding
from data_load import loadData, loadExperimentData
from easydict import EasyDict as edict
from tqdm import tqdm
import matplotlib.pylab as plt
import pdb

random.seed(1107)
torch.manual_seed(1107)

# def maxCriterion(element):
#     return len(element[0])
#
#
# def batchify(data, batch_size, classifier_mode, embedding_dim=1, randomize=True):
#     data = data
#     batches = list()
#     num_batches = len(data) // batch_size
#     leftover = len(data) - num_batches*batch_size
#     if randomize:
#         random.shuffle(data)
#     else:
#         data = data + data[:batch_size-leftover]
#         num_batches += 1
#     for b in range(num_batches):
#         # seqlen set to maximum of batch
#         batch = data[b*batch_size:(b+1)*batch_size]
#         batch = sorted(batch, key=maxCriterion, reverse=True)
#         dim = batch[0][0].shape[0]
#         real_lengths = [i[0].shape[0] for i in batch]
#         if embedding_dim == 1:
#             x_tensor = torch.zeros((batch_size, dim, embedding_dim), dtype=torch.long)
#         else:
#             x_tensor = torch.zeros((batch_size, dim, embedding_dim), dtype=torch.float)
#         y_event_tensor = torch.zeros((batch_size), dtype=torch.long)
#         y_crit_tensor = torch.zeros((batch_size), dtype=torch.long)
#         for i in range(batch_size):
#             x_i, y_i, y_cr = batch[i]
#             x_i = F.pad(x_i, (0, 0, 0, dim-x_i.shape[0]))
#             x_tensor[i] = x_i
#             y_event_tensor[i] = y_i
#             y_crit_tensor[i] = y_cr
#         if classifier_mode == 'event_type':
#             batches.append((x_tensor, y_event_tensor, real_lengths))
#         elif classifier_mode == 'criticality':
#             batches.append((x_tensor, y_crit_tensor, real_lengths))
#         elif classifier_mode in ['multitask', 'adversarial']:
#             batches.append((x_tensor, y_event_tensor, y_crit_tensor, real_lengths))
#         else:
#             batches = None
#     return batches


def visualize_saliency(gradients, tokens, figure_name, bar_width=10):
    matrix= gradients.numpy()
    matrix_magnify = np.zeros((matrix.shape[0] * bar_width, matrix.shape[1]))
    for i in range(matrix.shape[0]):
        for j in range(bar_width):
            matrix_magnify[i * bar_width + j, :] = matrix[i, :]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    extended_tokens=[]
    for token in tokens:
        extended_tokens+= [token]+ (bar_width-1)* ['']
    ax.set_yticks(np.arange(len(extended_tokens)))
    ax.set_yticklabels(extended_tokens)
    ax.xaxis.set_visible(False)
    plt.imshow(matrix_magnify, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.savefig('../figures/'+figure_name+'.pdf')
    #plt.show()

def get_embedding_activations(data, baseline_path, adv_path, data_path, desc_path, batch_size,
                    hidden_dim, embedding_type,
                    classifier_mode, event_type, optimizer_type,
                    num_layers, epochs, learning_rate, weight_decay, momentum, dropout, early_stop,
                    use_gpu, verbose ):

    train_data, val_data, events, vocab = loadExperimentData(desc_path=desc_path, embedding_type=embedding_type,
                                                   data_path=data_path)
    embedding_dim = train_data[0][0].shape[1]
    val_batches = batchify(val_data, batch_size, classifier_mode='adversarial', embedding_dim=embedding_dim, randomize=False)
    event_labels = events
    crit_labels = {'low': 0, 'high': 1}
    event_output_size = len(event_labels)
    crit_output_size = len(crit_labels)

    baseline_model = BiLSTM_BERT(embedding_dim=embedding_dim, hidden_dim=hidden_dim, label_size=crit_output_size,
                                 use_gpu=use_gpu, batch_size=batch_size, num_layers=num_layers, dropout=dropout)

    # baseline_model = torch.load(baseline_path)
    adv_model = torch.load(adv_path)

    # if optimizer_type == 'adam':
    #     optimizer_adv = optim.Adam(adv_model.parameters(), lr=learning_rate)
    #     optimizer_baseline = optim.Adam(baseline_model.parameters(), lr=learning_rate)
    # elif optimizer_type == 'adamw':
    #     optimizer_adv = optim.AdamW(adv_model.parameters(), lr=learning_rate)
    #     optimizer_baseline = optim.AdamW(baseline_model.parameters(), lr=learning_rate)
    # else:
    #     optimizer_adv = optim.SGD(adv_model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    #     optimizer_baseline = optim.SGD(baseline_model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

    #batches= batchify(data)

    for x, y_event, y_crit, seq_lengths, sentences in val_batches:
        adv_x= x.clone().detach().requires_grad_(True)
        adv_model.zero_grad()
        y_pred_event, y_pred_crit, _ = adv_model(adv_x, seq_lengths)
        loss_event = adv_model.loss(y_pred_event, y_event, seq_lengths, event_output_size)
        loss_crit = adv_model.loss(y_pred_crit, y_crit, seq_lengths, crit_output_size)
        loss = loss_event + loss_crit
        loss.backward()
        adv_gradients = adv_x.grad
        for i in range(batch_size):
            tokens= sentences[i]
            visualize_saliency(adv_gradients[i,:], tokens, 'figure_adv'+str(i))
        # optimizer_adv.step()
        del loss
        del loss_crit
        del loss_event
        baseline_x = x.clone().detach().requires_grad_(True)
        baseline_model.zero_grad()
        y_pred_baseline, _ = baseline_model(baseline_x, seq_lengths)
        #loss_event = baseline_model.loss(y_pred_event, y_event, seq_lengths, event_output_size)
        loss_baseline = baseline_model.loss(y_pred_baseline, y_crit, seq_lengths)
        loss_baseline.backward()
        baseline_gradients = baseline_x.grad
        for i in range(batch_size):
            tokens= sentences[i]
            visualize_saliency(baseline_gradients[i,:], tokens, 'figure_baseline'+str(i))
        # optimizer_baseline.step()
        del loss_baseline