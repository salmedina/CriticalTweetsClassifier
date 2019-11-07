import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', type=bool, default=False, help='Flag for using the GPU')
    parser.add_argument('--embedding_type', type=str, default='bert', help='Word embedding to be used: {glove, bert}')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size used during training')
    parser.add_argument('--hidden_dim', type=int, default=100, help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs for training the model')
    parser.add_argument('--lr', type=float, default=0.03, help='Training learning rate')
    parser.add_argument('--wd', type=float, default=0.03, help='Training weight decay')
    parser.add_argument('--data_path', type=str, help='Path to the json file to use for classification')
    parser.add_argument('--valid_freq', type=int, help='Number of epochs when the validation will be run')
    parser.add_argument('--output_path', type=str, default='./output',
                        help='Path to the directory where output will be saved')
    return parser.parse_args()

def main(args):
    #TODO: Implement the actual training and validation of the model
    pass

if __name__ == '__main__':
    args = parse_args()
    main(args)