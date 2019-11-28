import argparse
from classifier import train_model, train_multitask

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', type=bool, default=False, help='Flag for using the GPU')
    parser.add_argument('--embedding_type', type=str, default='bert', help='Word embedding to be used: {torch, glove, bert}')
    parser.add_argument('--event_type', type=str, default='earthquake',
                        help='Determines the subset of dataset used for experiment.')
    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='Word embedding dimension when using torch embeddings')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size used during training')
    parser.add_argument('--hidden_dim', type=int, default=100, help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for training the model')
    parser.add_argument('--task', type=str, default='criticality', help='Classification task: {criticality, event_type, multi_task}')
    parser.add_argument('--lr', type=float, default=0.03, help='Training learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='Training weight decay')
    parser.add_argument('--early_stop', type=bool, default=True, help='Enable/Disable early stopping based on F1')
    parser.add_argument('--data_path', type=str, help='Path to the json file to use for classification')
    parser.add_argument('--valid_freq', type=int, help='Number of epochs when the validation will be run')
    parser.add_argument('--output_path', type=str, default='./output',
                        help='Path to the directory where output will be saved')
    return parser.parse_args()


def print_train_params(args):
    print(f'''=== Training Params ==========================================
Event Type:     {args.event_type}
Embedding Type: {args.embedding_type}
Embedding Dim:  {args.embedding_dim}
Hidden Dim:     {args.hidden_dim}
Mode:           {args.task}
Num Layers:     {args.num_layers}
Batch Size:     {args.batch_size}
Learning Rate:  {args.lr}
Weight Decay:   {args.wd}
Early Stopping: {args.early_stop}
Use GPU:        {args.use_gpu}
==============================================================''')


def main(args):
    print_train_params(args)
    if args.task == 'multi_task':
        train_multitask(args.batch_size, args.hidden_dim, args.embedding_type, args.event_type,
                    args.num_layers, args.num_epochs, args.lr, args.wd, args.early_stop,
                    args.use_gpu)
    else:
        train_model(args.batch_size,
                    args.embedding_dim, args.hidden_dim, args.embedding_type,
                    args.task, args.event_type,
                    args.num_layers, args.num_epochs, args.lr, args.wd, args.early_stop,
                    args.use_gpu)

if __name__ == '__main__':
    args = parse_args()
    main(args)
