import json
import random
import math
import torch
import torch.optim as optim
from BiLSTM_Classifier import BiLSTMEventType
from BiLSTM_Static import BiLSTM_BERT, BiLSTM_BERT_MultiTask
import torch.nn.functional as F
from bert_embedding import BertEmbedding
from data_load import loadData
import pdb

random.seed(1107)
torch.manual_seed(1)


def maxCriterion(element):
    return len(element[0])


def batchify(data, batch_size, classifier_mode, embedding_dim=1, randomize=True):
    data = data
    batches = list()
    num_batches = len(data) // batch_size
    leftover = len(data) - num_batches*batch_size
    if randomize:
        random.shuffle(data)
    else:
        data = data + data[:batch_size-leftover]
        num_batches += 1
    for b in range(num_batches):
        batch = data[b*batch_size:(b+1)*batch_size]
        batch = sorted(batch, key= maxCriterion, reverse=True)
        dim = batch[0][0].shape[0]
        real_lengths = [i[0].shape[0] for i in batch]
        if embedding_dim == 1:
            x_tensor = torch.zeros((batch_size, dim, embedding_dim), dtype=torch.long)
        else:
            x_tensor = torch.zeros((batch_size, dim, embedding_dim), dtype=torch.float)

        y_event_tensor = torch.zeros((batch_size), dtype=torch.long)
        y_crit_tensor = torch.zeros((batch_size), dtype=torch.long)
        for i in range(batch_size):
            x_i, y_i, y_cr = batch[i]
            x_i = F.pad(x_i, (0, 0, 0, dim-x_i.shape[0]))
            x_tensor[i] = x_i
            y_event_tensor[i] = y_i
            y_crit_tensor[i] = y_cr

        if classifier_mode == 'criticality':
            batches.append((x_tensor, y_crit_tensor, real_lengths))
        elif classifier_mode == 'event_type':
            batches.append((x_tensor, y_event_tensor, real_lengths))
        elif classifier_mode == 'multi_task':
            batches.append((x_tensor, y_event_tensor, y_crit_tensor, real_lengths))
        else:
            batches = None

    return batches


def train_multitask(batch_size, hidden_dim, embedding_type, event_type,
                    num_layers, epochs, learning_rate, weight_decay,
                    use_gpu):

    print("Loading Data....")
    train_data, val_data, events, vocab = loadData(embedding_type,
                                         event_type=event_type)
    event_labels = events
    crit_labels = {'<PAD>': 0, 'low': 1, 'high': 2}
    event_label_size = len(event_labels)
    crit_label_size = len(crit_labels)

    print('Training multitask model...')
    if embedding_type == 'bert' or embedding_type == 'glove':
        embedding_dim = train_data[0][0].shape[1]
        val_data = batchify(val_data, batch_size, classifier_mode='multi_task', embedding_dim=embedding_dim, randomize=False)
        model = BiLSTM_BERT_MultiTask(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                            event_output_size=event_label_size, crit_output_size=crit_label_size,
                            use_gpu=use_gpu, batch_size=batch_size)
    else:
        print('Multi-task model only works with pre-trained embeddings')
        return

    if use_gpu:
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_f1 = 0.0
    for epoch in range(epochs):
        print('')
        print(f'======== Epoch Number: {epoch}')
        total_loss = 0.
        train_batches = batchify(train_data, batch_size, 'multi_task', embedding_dim=embedding_dim)

        for x, y_event, y_crit, seq_lengths in train_batches:
            model.zero_grad()
            y_pred_event, y_pred_crit = model(x, seq_lengths)
            loss_event = model.loss(y_pred_event, y_event, seq_lengths, event_label_size)
            loss_crit = model.loss(y_pred_crit, y_crit, seq_lengths, crit_label_size)
            loss = loss_event + loss_crit
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            del loss
            del loss_crit
            del loss_event

        # Validate the model
        with torch.no_grad():
            # Test on training data
            x, y_event, y_crit, seq_lengths = zip(*train_batches)
            train_event_batches = zip(x, y_event, seq_lengths)
            train_crit_batches = zip(x, y_crit, seq_lengths)
            print("Event Type ", event_type)
            accuracy, f1, final_metrics = test_event_type(model, train_event_batches, events)
            print(f"Event Train set - Acc: {accuracy:05f}    F1: {f1:05f}    Loss: {total_loss}")
            print(final_metrics)
            accuracy, f1, final_metrics = test_criticality(model, train_crit_batches, events)
            print(f"Crit. Train set - Acc: {accuracy:05f}    F1: {f1:05f}    Loss: {total_loss}")
            print(final_metrics)

            # Test on validation data
            x, y_event, y_crit, seq_lengths = zip(*val_data)
            val_event_data = zip(x, y_event, seq_lengths)
            val_crit_data = zip(x, y_crit, seq_lengths)
            accuracy, f1, final_metrics = test_event_type(model, val_event_data, events)
            print(f"Event Dev set - Acc: {accuracy:05f}    F1: {f1:05f}")
            accuracy, f1, final_metrics = test_criticality(model, val_crit_data, events)
            print(f"Crit. Dev set - Acc: {accuracy:05f}    F1: {f1:05f}")
            print(final_metrics)

            if f1 < best_f1:
                print('Early convergence. Training stopped.')
                print(f'Best F1: {best_f1}')
                break
            else:
                best_f1 = f1

    return model


def train_model(batch_size,
                embedding_dim, hidden_dim, embedding_type,
                classifier_mode, event_type,
                num_layers, epochs, learning_rate, weight_decay,
                use_gpu):

    print("Loading Data....")
    train, val, events, vocab = loadData(embedding_type, event_type=event_type)
    if classifier_mode == 'criticality':
        labels = {'<PAD>': 0, 'low': 1, 'high': 2}
    else:
        labels = events

    print('Training model...')
    if embedding_type == 'bert' or embedding_type == 'glove':
        embedding_dim = train[0][0].shape[1]
        val = batchify(val, batch_size, classifier_mode, embedding_dim=embedding_dim, randomize=False)
        model = BiLSTM_BERT(embedding_dim, hidden_dim, len(labels), use_gpu, batch_size, num_layers)
    else:
        val = batchify(val, batch_size, classifier_mode, randomize=False)
        model = BiLSTMEventType(embedding_dim, hidden_dim, len(vocab), len(labels), use_gpu, batch_size, num_layers)

    if use_gpu:
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_f1 = 0.0
    for epoch in range(epochs):
        print('')
        print(f'======== Epoch Number: {epoch}')
        total_loss = 0.
        if embedding_type == 'bert' or embedding_type == 'glove':
            train_i = batchify(train, batch_size, classifier_mode, embedding_dim= embedding_dim)
        else:
            train_i = batchify(train, batch_size, classifier_mode)
        for x, y, seq_lengths in train_i:
            model.zero_grad()
            y_pred = model(x, seq_lengths)
            loss = model.loss(y_pred, y, seq_lengths)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            del loss

        # Validate the model
        with torch.no_grad():
            if classifier_mode == 'criticality':
                accuracy, f1, final_metrics = test_criticality(model, train_i, events)
            else:
                accuracy, f1, final_metrics = test_event_type(model, train_i, events)
            print("Event Type ", event_type)
            print(f"Train set - Acc: {accuracy:05f}    F1: {f1:05f}    Loss: {total_loss}")
            print(final_metrics)
            if classifier_mode == 'criticality':
                accuracy, f1, final_metrics = test_criticality(model, val, events)
            else:
                accuracy, f1, final_metrics = test_event_type(model, val, events)
            print(f"Dev set - Acc: {accuracy:05f}    F1: {f1:05f}")
            print(final_metrics)
            if f1 < best_f1:
                print('Early convergence. Training stopped.')
                print(f'Best F1: {best_f1}')
                break
            else:
                best_f1 = f1

    return model


def test_criticality(model, data, events):
    correct = 0.0
    total = 0.0
    label_map = {1: 'low', 2: 'critical'}
    #event_scores= {event: {i:{'correct':0.0, 'gold':0.0001, 'predicted':0.0001} for i in label_map}for event in range(len(events))}
    criticality_scores= {i: {'correct': 0.0, 'gold': 0.0001, 'predicted': 0.0001} for i in label_map}
    for x, y, seq_lengths in data:
        total += len(y)
        y_pred = model(x, seq_lengths)
        y_pred_value = torch.argmax(y_pred, 1)
        vector = y-y_pred_value
        correct += (vector == 0).sum().item()
        for i in range(len(y)):
            actual = y[i].item()
            pred = y_pred_value[i].item()
            criticality_scores[actual]['gold'] += 1
            criticality_scores[pred]['predicted'] += 1
            if actual == pred:
                criticality_scores[actual]['correct'] += 1

    pr_low = criticality_scores[1]['correct'] / criticality_scores[1]['predicted']
    re_low = criticality_scores[1]['correct'] / criticality_scores[1]['gold']
    f1_low = (2 * pr_low * re_low) / (pr_low + re_low + 0.0001)

    pr_cr = criticality_scores[2]['correct'] / criticality_scores[2]['predicted']
    re_cr = criticality_scores[2]['correct'] / criticality_scores[2]['gold']
    f1_cr = (2*pr_cr*re_cr)/(pr_cr+re_cr+0.0001)

    accuracy = correct / total
    macro_f1 = (f1_cr+f1_low) / 2
    final_metrics = {'low': (pr_low, re_low, f1_low), 'critical': (pr_cr, re_cr, f1_cr)}

    return accuracy, macro_f1, final_metrics


def test_event_type(model, data, events):
    correct = 0.0
    total = 0.0
    event_scores = {event: {'correct': 0.0, 'gold': 0.0001, 'predicted': 0.0001} for event in range(len(events))}
    for x, y, seq_lengths in data:
        total += len(y)
        y_pred = model(x, seq_lengths)
        y_pred_value = torch.argmax(y_pred, 1)
        vector = y-y_pred_value
        correct += (vector == 0).sum().item()
        for i in range(len(y)):
            actual = y[i].item()
            pred = y_pred_value[i].item()
            event_scores[actual]['gold'] += 1
            event_scores[pred]['predicted'] += 1
            if actual == pred:
                event_scores[actual]['correct'] += 1
    final_metrics= {}
    macro_f1 = 0.0
    for event_id in event_scores:
        for e, v in events.items():
            if v == event_id:
                event = e
        pr = event_scores[event_id]['correct']/event_scores[event_id]['predicted']
        re = event_scores[event_id]['correct'] / event_scores[event_id]['gold']
        f1 = (2*pr*re)/(pr+re+0.0001)
        if event_id != 0:
            final_metrics[event] = (pr, re, f1)
            macro_f1 += f1
    accuracy = correct/total
    macro_f1 /= len(final_metrics)

    return accuracy, macro_f1, final_metrics


#train_model(16, 300, 100, 'bert', 'criticality', 'earthquake')
#train_model(16, 300, 100, 'bert', 'event', 'earthquake')