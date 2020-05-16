import os
import pickle
from linecache import getline
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from random import shuffle
import h5py

from common import Common
from config import Config


class Code2SeqDataset(Dataset):
    def __init__(self, data_file, config):
        super(Dataset, self).__init__()
        self.data_file = data_file
        self.config = config

        with open(self.config.DICT_PATH, 'rb') as f:
            self.subtoken_to_count = pickle.load(f)
            self.node_to_count = pickle.load(f)
            self.target_to_count = pickle.load(f)
            self.max_contexts = pickle.load(f)
            self.num_training_examples = pickle.load(f)
            print('Num training samples: {0}'.format(self.num_training_examples))
            print('Dictionaries loaded.')

        # Get subtoken vocab mapping
        self.subtoken_to_index, self.index_to_subtoken, self.subtoken_vocab_size = \
            Common.load_vocab_from_dict(self.subtoken_to_count, add_values=['<PAD>', '<UNK>'],
                                        max_size=self.config.SUBTOKENS_VOCAB_MAX_SIZE)
        print('Loaded subtoken vocab. size: %d' % self.subtoken_vocab_size)

        # Get target vocab mapping
        self.target_to_index, self.index_to_target, self.target_vocab_size = \
            Common.load_vocab_from_dict(self.target_to_count, add_values=['<PAD>', '<UNK>', '<S>'],
                                        max_size=self.config.TARGET_VOCAB_MAX_SIZE)
        print('Loaded target word vocab. size: %d' % self.target_vocab_size)

        # Get node vocab mapping
        self.node_to_index, self.index_to_node, self.nodes_vocab_size = \
            Common.load_vocab_from_dict(self.node_to_count, add_values=['<PAD>', '<UNK>'], max_size=None)
        print('Loaded nodes vocab. size: %d' % self.nodes_vocab_size)

        self.max_length_target = config.MAX_TARGET_PARTS
        self.max_length_leaf = config.MAX_NAME_PARTS
        self.max_length_ast_path = config.MAX_PATH_LENGTH

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, item):
        row = self.data_file[str(item)]["row"].value
        row = row.split()

        word = row[0]
        contexts = row[1:]
        shuffle(contexts)
        contexts = contexts[:config.MAX_CONTEXTS]

        start_leaf_matrix = torch.zeros(size=(config.MAX_CONTEXTS, self.max_length_leaf))
        ast_path_matrix = torch.zeros(size=(config.MAX_CONTEXTS, self.max_length_ast_path))
        end_leaf_matrix = torch.zeros(size=(config.MAX_CONTEXTS, self.max_length_leaf))
        target_matrix = torch.zeros(size=(config.MAX_CONTEXTS, self.max_length_target))
        for i, context in enumerate(contexts):
            leaf_node_1, ast_path, leaf_node_2 = context.split(',')
            if len(leaf_node_1) > self.max_length_leaf:
                leaf_node_1 = leaf_node_1[:self.max_length_leaf]
            if len(ast_path) > self.max_length_ast_path:
                ast_path = ast_path[:self.max_length_ast_path]
            if len(leaf_node_2) > self.max_length_leaf:
                leaf_node_2 = leaf_node_2[:self.max_length_leaf]
            if len(word) > self.max_length_target:
                word = word[:self.max_length_target]

            target = [self.target_to_index.get(w, self.target_to_index['<UNK>']) for w in word.split('|')]
            ast_path = [self.node_to_index.get(w, self.node_to_index['<UNK>']) for w in ast_path.split('|')]
            leaf_node_1 = [self.subtoken_to_index.get(w, self.subtoken_to_index['<UNK>']) for w in leaf_node_1.split('|')]
            leaf_node_2 = [self.subtoken_to_index.get(w, self.subtoken_to_index['<UNK>']) for w in leaf_node_2.split('|')]

            start_leaf_matrix[i, :len(leaf_node_1)] = torch.tensor(leaf_node_1)
            ast_path_matrix[i, :len(ast_path)] = torch.tensor(ast_path)
            end_leaf_matrix[i, :len(leaf_node_2)] = torch.tensor(leaf_node_2)
            target_matrix[i, :len(target)] = torch.tensor(target)

        return start_leaf_matrix, ast_path_matrix, end_leaf_matrix, target_matrix


if __name__ == "__main__":
    config = Config.get_default_config(None)

    test_set = Code2SeqDataset(h5py.File((config.H5_FOLDER / "test.h5"), mode='r'), config=config)
    train_set = Code2SeqDataset(h5py.File((config.H5_FOLDER / "train.h5"), mode='r'), config=config)
    val_set = Code2SeqDataset(h5py.File((config.H5_FOLDER / "val.h5"), mode='r'), config=config)

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    for (start_leaf, ast_path, end_leaf, target) in train_loader:
        print(start_leaf.shape)
        print(ast_path.shape)
        print(end_leaf.shape)
        print(target.shape)
        print(" ".join([test_set.index_to_target.get(x.item(), '??') for x in target[0, 0, :]]))
        break

