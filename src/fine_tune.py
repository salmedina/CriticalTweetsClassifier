import argparse
import numpy as np
from classifier import train_model, train_multitask

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', type=int, default=20, help='Number of random hyper-param sets to be tested')
    parser.add_argument('--use_gpu', type=bool, default=False, help='Flag for using the GPU')
    parser.add_argument('--embedding_type', type=str, default='bert', help='Word embedding to be used: {torch, glove, bert}')
    parser.add_argument('--event_type', type=str, default='earthquake',
                        help='Determines the subset of dataset used for experiment. If multiple, separate them with commas')
    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='Word embedding dimension when using torch embeddings')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size used during training')
    parser.add_argument('--hidden_dim', type=int, default=100, help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs for training the model')
    parser.add_argument('--task', type=str, default='criticality', help='Classification task: {criticality, event_type, multitask, adversarial}')
    parser.add_argument('--early_stop', action='store_true', default=False, help='Enable/Disable early stopping based on F1')
    parser.add_argument('--data_path', type=str, default='../data/labeled_data.json', help='Path to the json file to use for classification')
    parser.add_argument('--exp_desc', type=str, default=None, help='Path to the experiment description yaml file')
    parser.add_argument('--output_path', type=str, default='./output',
                        help='Path to the directory where output will be saved')
    parser.add_argument('--mute', action='store_false', default=True, help='Print test results on every epoch')

    return parser.parse_args()


def is_valid_task(task):
    return task in ['criticality', 'multitask', 'event_type', 'adversarial']


def print_train_params(args):
    print(f'''=== Training Params ==========================================
    Embedding Type:    {args.embedding_type}
    Embedding Dim:     {args.embedding_dim}
    Hidden Dim:        {args.hidden_dim}
    Mode:              {args.task}
    Num Layers:        {args.num_layers}
    Batch Size:        {args.batch_size}
    Early Stopping:    {args.early_stop}
    Exp. Descriptor:   {args.exp_desc}
    Use GPU:           {args.use_gpu}
==============================================================''')


def main(args):
    print_train_params(args)
    if not is_valid_task(args.task):
        print(f'{args.task} is an invalid task')
        return

    for run_id in range(args.num_runs):
        print(f'EXPERIMENT {run_id}')
        lr = np.exp(np.random.uniform(np.log(1e-5), np.log(1.)))
        wd = np.exp(np.random.uniform(np.log(1e-7), np.log(1.)))
        m = np.random.uniform(0.01, 0.99)

        if args.task in ['multitask', 'adversarial']:
            train_multitask(args.data_path, args.exp_desc, args.batch_size,
                            args.hidden_dim, args.embedding_type,
                            args.task, args.event_type,
                            args.num_layers, args.num_epochs, lr, wd, m, args.early_stop,
                            args.use_gpu, args.mute)
        elif args.task in ['event_type', 'criticality']:
            train_model(args.data_path, args.exp_desc, args.batch_size,
                        args.embedding_dim, args.hidden_dim, args.embedding_type,
                        args.task, args.event_type,
                        args.num_layers, args.num_epochs, lr, wd, m, args.early_stop,
                        args.use_gpu, args.mute)

        print(f'LR: {lr}     WD: {wd}     M: {m}')
        print('-----------------------------------------------------------------')


if __name__ == '__main__':
    args = parse_args()
    main(args)
