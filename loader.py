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
        data_file += ".h5"
        self.data_file = h5py.File((config.H5_FOLDER / data_file), mode='r')
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

        self.max_length_target = self.config.MAX_TARGET_PARTS
        self.max_length_leaf = self.config.MAX_NAME_PARTS
        self.max_length_ast_path = self.config.MAX_PATH_LENGTH

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, item):
        row = self.data_file[str(item)]["row"][()]
        row = row.split()

        word = row[0]
        contexts = row[1:]
        shuffle(contexts)
        contexts = contexts[:self.config.MAX_CONTEXTS]

        # Initialise matrices
        start_leaf_matrix = torch.zeros(size=(self.config.MAX_CONTEXTS, self.max_length_leaf))
        ast_path_matrix = torch.zeros(size=(self.config.MAX_CONTEXTS, self.max_length_ast_path))
        end_leaf_matrix = torch.zeros(size=(self.config.MAX_CONTEXTS, self.max_length_leaf))
        target_vector = torch.zeros(size=(self.max_length_target,))

        start_leaf_mask = torch.zeros(size=(self.config.MAX_CONTEXTS, self.max_length_leaf))
        end_leaf_mask = torch.zeros(size=(self.config.MAX_CONTEXTS, self.max_length_leaf))
        target_mask = torch.zeros(size=(self.max_length_target,))

        context_mask = torch.zeros(size=(self.config.MAX_CONTEXTS, ))
        ast_path_lengths = torch.zeros(size=(self.config.MAX_CONTEXTS, ))

        for i, context in enumerate(contexts):
            leaf_node_1, ast_path, leaf_node_2 = context.split(',')

            leaf_node_1 = leaf_node_1[:self.max_length_leaf]
            ast_path = ast_path[:self.max_length_ast_path]
            leaf_node_2 = leaf_node_2[:self.max_length_leaf]

            ast_path = [self.node_to_index.get(w, self.node_to_index['<UNK>']) for w in ast_path.split('|')]
            leaf_node_1 = [self.subtoken_to_index.get(w, self.subtoken_to_index['<UNK>']) for w in leaf_node_1.split('|')]
            leaf_node_2 = [self.subtoken_to_index.get(w, self.subtoken_to_index['<UNK>']) for w in leaf_node_2.split('|')]

            start_leaf_matrix[i, :len(leaf_node_1)] = torch.tensor(leaf_node_1)
            start_leaf_mask[i, :len(leaf_node_1)] = torch.ones(size=(len(leaf_node_1),))

            ast_path_matrix[i, :len(ast_path)] = torch.tensor(ast_path)
            ast_path_lengths[i] = torch.tensor(len(ast_path))

            end_leaf_matrix[i, :len(leaf_node_2)] = torch.tensor(leaf_node_2)
            end_leaf_mask[i, :len(leaf_node_2)] = torch.ones(size=(len(leaf_node_2),))

        context_mask[:len(contexts)] = torch.ones(size=(len(contexts),))

        word = word[:self.max_length_target]
        target = [self.target_to_index.get(w, self.target_to_index['<UNK>']) for w in word.split('|')]
        target_vector[:len(target)] = torch.tensor(target)
        target_mask[:len(target)] = torch.ones(size=(len(target),))

        return (start_leaf_matrix, ast_path_matrix, end_leaf_matrix, target_vector,
                start_leaf_mask, end_leaf_mask, target_mask, context_mask,
                ast_path_lengths)


if __name__ == "__main__":
    config = Config.get_default_config(None)

    test_set = Code2SeqDataset('test', config=config)
    train_set = Code2SeqDataset('train', config=config)
    val_set = Code2SeqDataset('val', config=config)

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    for train_data in train_loader:
        print("start_leaf_matrix", train_data[0].shape)
        print("ast_path_matrix", train_data[1].shape)
        print("end_leaf_matrix", train_data[2].shape)
        print("target_vector", train_data[3].shape)
        print("start_leaf_mask", train_data[4].shape)
        print("end_leaf_mask", train_data[5].shape)
        print("target_mask", train_data[6].shape)
        print("context_mask", train_data[7].shape)
        print("ast_path_lengths", train_data[8].shape)

        print(" ".join([test_set.index_to_target.get(x.item(), '<UNK>') for x in train_data[3][0]]))
        break

