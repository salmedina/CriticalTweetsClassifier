import random
import torch
import torch.optim as optim
import numpy as np
from BiLSTM_Classifier import BiLSTM_Baseline
from BiLSTM_Static import BiLSTM_BERT, BiLSTM_BERT_MultiTask, BiLSTM_BERT_Adversarial
import torch.nn.functional as F
from bert_embedding import BertEmbedding
from data_load import loadData, loadExperimentData
from easydict import EasyDict as edict
from tqdm import tqdm
import pdb

random.seed(1107)
torch.manual_seed(1107)


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
        # seqlen set to maximum of batch
        sentences=[]
        batch = data[b*batch_size:(b+1)*batch_size]
        batch = sorted(batch, key=maxCriterion, reverse=True)
        dim = batch[0][0].shape[0]
        real_lengths = [i[0].shape[0] for i in batch]
        if embedding_dim == 1:
            x_tensor = torch.zeros((batch_size, dim, embedding_dim), dtype=torch.long)
        else:
            x_tensor = torch.zeros((batch_size, dim, embedding_dim), dtype=torch.float)

        y_event_tensor = torch.zeros((batch_size), dtype=torch.long)
        y_crit_tensor = torch.zeros((batch_size), dtype=torch.long)
        for i in range(batch_size):
            x_i, y_i, y_cr, text = batch[i]
            x_i = F.pad(x_i, (0, 0, 0, dim-x_i.shape[0]))
            x_tensor[i] = x_i
            y_event_tensor[i] = y_i
            y_crit_tensor[i] = y_cr
            sentences.append(text)
        if classifier_mode == 'event_type':
            batches.append((x_tensor, y_event_tensor, real_lengths, sentences))
        elif classifier_mode == 'criticality':
            batches.append((x_tensor, y_crit_tensor, real_lengths, sentences))
        elif classifier_mode in ['multitask', 'adversarial']:
            batches.append((x_tensor, y_event_tensor, y_crit_tensor, real_lengths, sentences))
        else:
            batches = None

    return batches


def train_multitask(data_path, desc_path, batch_size,
                    hidden_dim, embedding_type,
                    classifier_mode, event_type, optimizer_type,
                    num_layers, epochs, learning_rate, weight_decay, momentum, dropout, early_stop,
                    use_gpu, verbose, output_path):

    print("Loading Data....")
    if desc_path is None:
        train_data, val_data, events, vocab = loadData(embedding_type, data_path=data_path, event_type=event_type)
    else:
        train_data, val_data, events, vocab = loadExperimentData(desc_path=desc_path,
                                                   embedding_type=embedding_type,
                                                   data_path=data_path)
    event_labels = events
    crit_labels = {'low': 0, 'high': 1}
    event_output_size = len(event_labels)
    crit_output_size = len(crit_labels)

    print(f'Training {classifier_mode} model...')
    if embedding_type in ['bert', 'glove']:
        embedding_dim = train_data[0][0].shape[1]
        val_data = batchify(val_data, batch_size, classifier_mode='multitask', embedding_dim=embedding_dim, randomize=False)
        if classifier_mode == 'multitask':
            model = BiLSTM_BERT_MultiTask(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                                          event_output_size=event_output_size, crit_output_size=crit_output_size,
                                          use_gpu=use_gpu, batch_size=batch_size, dropout=dropout)
        elif classifier_mode == 'adversarial':
            model = BiLSTM_BERT_Adversarial(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                                            event_output_size=event_output_size, crit_output_size=crit_output_size,
                                            use_gpu=use_gpu, batch_size=batch_size, dropout=dropout)
        else:
            model = None
            print('Unknown classifier mode')

    else:
        #TODO: Implement multi-task learning for learning embeddings
        print('ERROR: Multi-task model only works with pre-trained embeddings')
        return

    if use_gpu:
        model = model.cuda()

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)


    best = edict(epoch=0, acc=0.0, f1=0.0, critical_f1=0.0, class_metrics=None)
    for epoch in tqdm(range(epochs)):
        if verbose:
            print('')
            print(f'======== Epoch Number: {epoch}')
        total_loss = 0.
        train_batches = batchify(train_data, batch_size, 'multitask', embedding_dim=embedding_dim)

        for x, y_event, y_crit, seq_lengths, sentences in train_batches:
            model.zero_grad()
            y_pred_event, y_pred_crit, _ = model(x, seq_lengths)
            loss_event = model.loss(y_pred_event, y_event, seq_lengths, event_output_size)
            loss_crit = model.loss(y_pred_crit, y_crit, seq_lengths, crit_output_size)
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
            if verbose:
                test_res = test_multitask(model=model, data=train_batches,
                                          event_labels_dict=event_labels,
                                          crit_labels_dict=crit_labels)
                print("Event Type ", event_type)
                print('⭑ Train Set ⭑')
                print(f"Event - Acc: {test_res.event.accuracy:05f}    F1: {test_res.event.f1:05f}    Loss: {total_loss}")
                print(test_res.event.final_metrics)
                print(f"Crit. - Acc: {test_res.crit.accuracy:05f}    F1: {test_res.crit.f1:05f}    Loss: {total_loss}")
                print(test_res.crit.final_metrics)

            # Test on validation data
            test_res = test_multitask(model=model, data=val_data,
                                      event_labels_dict=event_labels,
                                      crit_labels_dict=crit_labels)
            if verbose:
                print('⭑ Val Set ⭑')
                print(f"Event - Acc: {test_res.event.accuracy:05f}    F1: {test_res.event.f1:05f}    Loss: {total_loss}")
                print(test_res.event.final_metrics)
                print(f"Crit. - Acc: {test_res.crit.accuracy:05f}    F1: {test_res.crit.f1:05f}    Loss: {total_loss}")
                print(test_res.crit.final_metrics)

            critical_f1 = test_res.crit.final_metrics['high'][2]
            if (critical_f1 < best.critical_f1) and early_stop:
                print('Early convergence. Training stopped.')
                break
            elif critical_f1 > best.critical_f1:
                best.epoch = epoch
                best.acc = test_res.crit.accuracy
                best.f1 = test_res.crit.f1
                best.critical_f1 = test_res.crit.final_metrics['high'][2]
                best.class_metrics = test_res.crit.final_metrics
                torch.save(dict(mode=classifier_mode, 
                                model_state_dict=model.state_dict(),
                                optimizer_state_dict=optimizer.state_dict(),
                                optimizer_params=dict(),
                                ctor_params=dict(embedding_dim=embedding_dim, 
                                                hidden_dim=hidden_dim, 
                                                num_layers=num_layers,
                                                event_output_size=event_output_size, 
                                                crit_output_size=crit_output_size,
                                                use_gpu=use_gpu, 
                                                batch_size=batch_size, 
                                                dropout=dropout)),
                            output_path)

    print(f'{classifier_mode} {embedding_type}')
    if best.epoch > 0:
        print(f'''Best model:
        Epoch:   {best.epoch}
        Acc:     {best.acc:04f}
        F1:      {best.f1:04f}
        Metrics: {best.class_metrics}''')
        print(f'{best.epoch} {best.acc} {best.f1} {best.class_metrics["low"][2]} {best.class_metrics["high"][2]}')
    else:
        print('No convergence')

    return model


def train_model(data_path, desc_path, batch_size,
                embedding_dim, hidden_dim, embedding_type,
                classifier_mode, event_type, optimizer_type,
                num_layers, epochs, learning_rate, weight_decay, momentum, dropout, early_stop,
                use_gpu, verbose, output_path):

    print("Loading Data....")
    if desc_path is None:
        train, val, events, vocab = loadData(embedding_type, data_path=data_path, event_type=event_type)
    else:
        train, val, events, vocab = loadExperimentData(desc_path=desc_path,
                                                       embedding_type=embedding_type,
                                                       data_path=data_path)
    if classifier_mode == 'criticality':
        labels_dict = {'low': 0, 'high': 1}
    else:
        labels_dict = events

    print('Training model...')
    if embedding_type in ['bert', 'glove']:
        embedding_dim = train[0][0].shape[1]
        val = batchify(val, batch_size, classifier_mode, embedding_dim=embedding_dim, randomize=False)
        model = BiLSTM_BERT(embedding_dim=embedding_dim, hidden_dim=hidden_dim, label_size=len(labels_dict), use_gpu=use_gpu, batch_size=batch_size, num_layers=num_layers, dropout=dropout)
    else:
        val = batchify(val, batch_size, classifier_mode, randomize=False)
        model = BiLSTM_Baseline(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=len(vocab), label_size=len(labels_dict), use_gpu=use_gpu, batch_size=batch_size, num_layers=num_layers, dropout=dropout)

    if use_gpu:
        model = model.cuda()

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

    best = edict(epoch=0, acc=0.0, f1=0.0, critical_f1=0.0, class_metrics=None)
    for epoch in tqdm(range(epochs)):
        if verbose:
            print('')
            print(f'======== Epoch Number: {epoch}')
        total_loss = 0.
        if embedding_type in ['bert', 'glove']:
            train_i = batchify(train, batch_size, classifier_mode, embedding_dim= embedding_dim)
        else:
            train_i = batchify(train, batch_size, classifier_mode)

        for x, y, seq_lengths, sentences in train_i:
            model.zero_grad()
            y_pred, _ = model(x, seq_lengths)
            loss = model.loss(y_pred, y, seq_lengths)
            total_loss += loss.item()
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del loss

        # Validate the model
        with torch.no_grad():
            if verbose:
                accuracy, f1, final_metrics = test_model(model, train_i, labels_dict)
                print("Event Type ", event_type)
                print(f"Train set - Acc: {accuracy:05f}    F1: {f1:05f}    Loss: {total_loss}")
                print(final_metrics)

            accuracy, f1, final_metrics = test_model(model, val, labels_dict)
            if verbose:
                print(f"Dev set - Acc: {accuracy:05f}    F1: {f1:05f}")
                print(final_metrics)

            critical_f1 = final_metrics['high'][2]
            if (critical_f1 < best.critical_f1) and early_stop:
                print('Early convergence. Training stopped.')
                break
            elif critical_f1 > best.critical_f1:
                best.epoch = epoch
                best.acc = accuracy
                best.f1 = f1
                best.critical_f1 = critical_f1
                best.class_metrics = final_metrics
                torch.save(dict(mode='baseline', 
                                model_state_dict=model.state_dict(),
                                optimizer_state_dict=optimizer.state_dict(),
                                ctor_params=dict(embedding_dim=embedding_dim, 
                                                hidden_dim=hidden_dim, 
                                                label_size=len(labels_dict), 
                                                use_gpu=use_gpu, 
                                                batch_size=batch_size, 
                                                num_layers=num_layers, 
                                                dropout=dropout)),
                            output_path)

    print(f'{classifier_mode} {embedding_type}')
    if best.epoch > 0:
        print(f'''Best model:
        Epoch:   {best.epoch}
        Acc:     {best.acc:04f}
        F1:      {best.f1:04f}
        Metrics: {best.class_metrics}''')
        print(f'{best.epoch} {best.acc} {best.f1} {best.class_metrics["low"][2]} {best.class_metrics["high"][2]}')
    else:
        print('No convergence')

    return model


def invert_dict(input_dict):
    return {input_dict[key]: key for key in input_dict}


def update_pred_scores(y, y_pred, scores):
    y_pred_value = torch.argmax(y_pred, 1)
    diff_vector = y - y_pred_value
    correct = (diff_vector == 0).sum().item()
    for i in range(len(y)):
        actual = y[i].item()
        pred = y_pred_value[i].item()
        scores[actual]['gold'] += 1
        scores[pred]['predicted'] += 1
        if actual == pred:
            scores[actual]['correct'] += 1

    return scores, correct


def calc_metrics(scores, label_map):
    final_metrics = {}
    macro_f1 = 0.0
    for label_idx in scores:
        precision = scores[label_idx]['correct'] / scores[label_idx]['predicted']
        recall = scores[label_idx]['correct'] / scores[label_idx]['gold']
        f1 = (2 * precision * recall) / (precision + recall + 1e-4)
        label = label_map[label_idx]
        final_metrics[label] = (precision, recall, f1)
        macro_f1 += f1

    macro_f1 /= len(final_metrics)

    return macro_f1, final_metrics


def test_model(model, data, labels_dict):
    correct = 0.0
    total = 0.0

    all_pred, all_y = list(), list()
    all_embeddings = list()
    label_map = invert_dict(labels_dict)
    scores = {label_idx: {'correct': 0.0, 'gold': 0.0001, 'predicted': 0.0001} for label_idx in label_map}
    for x, y, seq_lengths, sentences in data:
        total += len(y)
        y_pred, embeddings = model(x, seq_lengths)
        y_pred_value = torch.argmax(y_pred, 1)
        all_pred += y_pred_value.tolist()
        all_y += y.tolist()
        all_embeddings.append(embeddings)
        diff_vector = y - y_pred_value
        correct += (diff_vector == 0).sum().item()
        for i in range(len(y)):
            actual = y[i].item()
            pred = y_pred_value[i].item()
            scores[actual]['gold'] += 1
            scores[pred]['predicted'] += 1
            if actual == pred:
                scores[actual]['correct'] += 1

    embeddings = np.concatenate(all_embeddings, axis=0)
    accuracy = correct / total
    macro_f1, final_metrics = calc_metrics(scores=scores, label_map=label_map)

    return accuracy, macro_f1, final_metrics


def test_multitask(model, data, event_labels_dict, crit_labels_dict):
    correct = edict(event=0.0, crit=0.0)
    total = 0.0

    label_map_event = invert_dict(event_labels_dict)
    label_map_crit = invert_dict(crit_labels_dict)

    scores_event = {label_idx: {'correct': 0.0, 'gold': 0.0001, 'predicted': 0.0001} for label_idx in label_map_event}
    scores_crit = {label_idx: {'correct': 0.0, 'gold': 0.0001, 'predicted': 0.0001} for label_idx in label_map_crit}

    all_scores, all_pred, all_y = list(), list(), list()
    all_embeddings = list()
    for x, y_event, y_crit, seq_lengths, sentences in data:
        total += len(y_event)
        y_pred_event, y_pred_crit, embeddings = model(x, seq_lengths)
        all_scores.append(torch.exp(y_pred_crit).numpy())

        #TODO: used for qualitative analysis, need to make it callable
        y_pred_crit_value = torch.argmax(y_pred_crit, 1)
        all_pred += y_pred_crit_value.tolist()
        all_y += y_crit.tolist()
        all_embeddings.append(embeddings)

        scores_event, correct_batch = update_pred_scores(y_event, y_pred_event, scores_event)
        correct.event += correct_batch

        scores_crit, correct_batch = update_pred_scores(y_crit, y_pred_crit, scores_crit)
        correct.crit += correct_batch

    all_scores = np.concatenate(all_scores)

    accuracy_event = correct.event / total
    macro_f1_event, final_metrics_event = calc_metrics(scores=scores_event, label_map=label_map_event)

    accuracy_crit = correct.crit / total
    macro_f1_crit, final_metrics_crit = calc_metrics(scores=scores_crit, label_map=label_map_crit)

    return edict(event=edict(accuracy=accuracy_event, f1=macro_f1_event, final_metrics=final_metrics_event),
                 crit=edict(accuracy=accuracy_crit, f1=macro_f1_crit, final_metrics=final_metrics_crit))

def load_model(model_path):
    ckpt = torch.load(model_path)
    params = edict(ckpt['ctor_params'])
    if ckpt['mode'] == 'baseline':
        model = BiLSTM_BERT(embedding_dim=params.embedding_dim, 
                            hidden_dim=params.hidden_dim, 
                            label_size=params.label_size, 
                            use_gpu=params.use_gpu, 
                            batch_size=params.batch_size, 
                            num_layers=params.num_layers, 
                            dropout=params.dropout)
    elif ckpt['mode'] == 'multitask':
        model = BiLSTM_BERT_MultiTask(embedding_dim=params.embedding_dim, 
                            hidden_dim=params.hidden_dim, 
                            num_layers=params.num_layers,
                            event_output_size=params.event_output_size, 
                            crit_output_size=params.crit_output_size,
                            use_gpu=params.use_gpu, 
                            batch_size=params.batch_size, 
                            dropout=params.dropout)
    elif ckpt['mode'] == 'adversarial':
        model = BiLSTM_BERT_Adversarial(embedding_dim=params.embedding_dim, 
                            hidden_dim=params.hidden_dim, 
                            num_layers=params.num_layers,
                            event_output_size=params.event_output_size, 
                            crit_output_size=params.crit_output_size,
                            use_gpu=params.use_gpu, 
                            batch_size=params.batch_size, 
                            dropout=params.dropout)
    else:
        model = None
    
    model.load_state_dict(ckpt['model_state_dict'])

    return model

