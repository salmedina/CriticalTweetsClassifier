import argparse
from classifier import train_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', type=bool, default=False, help='Flag for using the GPU')
    parser.add_argument('--embedding_type', type=str, default='torch', help='Word embedding to be used: {torch, glove, bert}')
    parser.add_argument('--event_type', type=str, default='earthquake',
                        help='Determines the subset of dataset used for experiment.')
    parser.add_argument('--embedding_dim', type=int, default='300',
                        help='Word embedding dimension when using torch embeddings')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size used during training')
    parser.add_argument('--hidden_dim', type=int, default=100, help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for training the model')
    parser.add_argument('--task', type=str, default='criticality', help='Classification task: {criticality, event_type}')
    parser.add_argument('--lr', type=float, default=0.03, help='Training learning rate')
    parser.add_argument('--wd', type=float, default=0.03, help='Training weight decay')
    parser.add_argument('--data_path', type=str, help='Path to the json file to use for classification')
    parser.add_argument('--valid_freq', type=int, help='Number of epochs when the validation will be run')
    parser.add_argument('--output_path', type=str, default='./output',
                        help='Path to the directory where output will be saved')
    return parser.parse_args()

def main(args):
    train_model(args.batch_size,
                args.embedding_dim, args.hidden_dim, args.embedding_type,
                args.task, args.event_type,
                args.num_layers, args.num_epochs, args.use_gpu)

if __name__ == '__main__':
    args = parse_args()
    main(args)
