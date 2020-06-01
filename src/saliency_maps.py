import random
import torch
import torch.optim as optim
import numpy as np
from classifier import batchify, maxCriterion, load_model
import torch.nn.functional as F
from bert_embedding import BertEmbedding
from data_load import loadData, loadExperimentData
from easydict import EasyDict as edict
from tqdm import tqdm
import matplotlib.pylab as plt
import pdb

random.seed(1107)
torch.manual_seed(1107)



def visualize_saliency(gradients, tokens, ax, bar_width=10):
    matrix= gradients.numpy()
    matrix_magnify = np.zeros((matrix.shape[0] * bar_width, matrix.shape[1]))
    for i in range(matrix.shape[0]):
        for j in range(bar_width):
            matrix_magnify[i * bar_width + j, :] = matrix[i, :]
    extended_tokens=[]
    for token in tokens:
        extended_tokens+=  3* ['']+[token]+ (bar_width-4)* ['']
    ax.set_yticks(np.arange(len(extended_tokens)))
    ax.set_yticklabels(extended_tokens)
    ax.xaxis.set_visible(False)
    img = ax.imshow(matrix_magnify, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(img, ax=ax)


def get_embedding_activations(baseline_path, adv_path, data_path, desc_path, batch_size,
                   embedding_type):

    train_data, val_data, events, vocab = loadExperimentData(desc_path=desc_path, embedding_type=embedding_type,
                                                  data_path=data_path)

    embedding_dim = train_data[0][0].shape[1]
    val_batches = batchify(val_data, batch_size, classifier_mode='adversarial', embedding_dim=embedding_dim, randomize=False)
    crit_labels = {'low': 0, 'high': 1}
    crit_output_size = len(crit_labels)

    baseline_model= load_model(baseline_path)
    adv_model = load_model(adv_path)
    index=0
    for x, y_event, y_crit, seq_lengths, sentences in val_batches:
        adv_x= x.clone().detach().requires_grad_(True)
        adv_model.zero_grad()
        y_pred_event, y_pred_crit, _ = adv_model(adv_x, seq_lengths)
        loss = adv_model.loss(y_pred_crit, y_crit, seq_lengths, crit_output_size)
        #loss = loss_event + loss_crit
        #loss= loss_crit
        loss.backward()
        adv_gradients = adv_x.grad
        del loss
        #del loss_crit
        baseline_x = x.clone().detach().requires_grad_(True)
        baseline_model.zero_grad()
        y_pred_baseline, _ = baseline_model(baseline_x, seq_lengths)
        loss_baseline = baseline_model.loss(y_pred_baseline, y_crit, seq_lengths)
        loss_baseline.backward()
        baseline_gradients = baseline_x.grad
        for i in range(batch_size):
            print('Current Sentence: '+str(index) +'\t'+str(y_crit[i].item())+ '\t'+ sentences[i])
            index+=1
            tokens= sentences[i].split()

            fig, (ax1, ax2) = plt.subplots(1, 2)
            #fig.set_figwidth(50)
            fig.set_figwidth(9)
            ax1.title.set_text('Baseline Model')
            ax2.title.set_text('Adversarial Model')
            visualize_saliency(baseline_gradients[i, :seq_lengths[i]], tokens, ax1)
            visualize_saliency(adv_gradients[i,:seq_lengths[i]], len(tokens)*[''], ax2)
            fig.savefig('../figures/figure_' +str(index)+ '.pdf')

        del loss_baseline





desc_path='../data/labeled_data.json'
embedding_type= 'glove'
data_path= '../data/experiments/flood_only_1.yaml'


get_embedding_activations('../models/flood_1_base_glove.pth', '../models/flood_1_adv_glove.pth', desc_path, data_path, 16,
                    'glove')