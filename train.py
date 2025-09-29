import torch
import numpy as np
import random
import argparse
import logging
from model.trainer import trainer
from model.graph_moe import GNNWithMoE
from model.data_loader import load_bpf_datasets


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for initialization')

    parser.add_argument('--dataset_path', type=str, default='./output', 
                        help='The base dataset path')

    parser.add_argument('--vul_type', type=str, default='')

    parser.add_argument('--glove_path', type=str, default='./embeds/glove.6B.300d.txt',  
                        help='The glove embedding path')

    parser.add_argument('--input_dim', type=int, default=300,
                        help='Dimension of input embeddings')

    parser.add_argument('--hidden_dim', type=int, default=300,
                        help='Dimension of hidden embeddings')

    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")

    parser.add_argument("--test_batch_size", default=128, type=int,  
                        help="Batch size per GPU/CPU for testing.")

    parser.add_argument("--lr", default=1e-3, type=float,
                        help="The initial learning rate for Adam. ")

    parser.add_argument('--epochs', type=int, default=50,
                        help='Total number of training epochs to perform')

    parser.add_argument('--test_only', type=bool, default=False,
                        help='Whether to run the test only')

    parser.add_argument('--test_mode', type=str, default="common",  
                        help='Choose to run train/test mode or directly scan the online contracts')

    parser.add_argument('--online_dataset_path', type=str, default="./online_output",  # online
                        help='Path to online dataset')

    parser.add_argument('--partial', type=str, default="0", 
                        help='Test online contract')

    return parser.parse_args()


def check_args(args):
    logger.info(vars(args))


def main():
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # Parse args
    args = parse_args()

    # Setup CUDA, GPU training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.device = device
    logger.info('Device is %s', args.device)

    # Set seed
    set_seed(args)

    # Print args
    check_args(args)

    # Load datasets
    train_dataset, test_dataset, file_label_maps = load_bpf_datasets(args)

    # Build model
    model = GNNWithMoE(num_node_features=args.input_dim, hidden_dim=300, num_classes=2, num_experts=4, dropout=0.1).to(device)
    logger.info('Build model successfully! ')

    # Train
    trainer(model, train_dataset, test_dataset, file_label_maps, args)

    print('Done! ')


if __name__ == "__main__":
    main()
