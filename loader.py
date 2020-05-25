import os
import pickle
from linecache import getline
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from random import shuffle
import h5py
import numpy as np

from common import Common
from config import Config


class Dictionaries:
    def __init__(self, config):
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
            Common.load_vocab_from_dict(self.subtoken_to_count, add_values=[Common.PAD, Common.UNK],
                                        max_size=self.config.SUBTOKENS_VOCAB_MAX_SIZE)
        print('Loaded subtoken vocab. size: %d' % self.subtoken_vocab_size)

        # Get target vocab mapping
        self.target_to_index, self.index_to_target, self.target_vocab_size = \
            Common.load_vocab_from_dict(self.target_to_count, add_values=[Common.PAD, Common.UNK, Common.SOS],
                                        max_size=self.config.TARGET_VOCAB_MAX_SIZE)
        print('Loaded target word vocab. size: %d' % self.target_vocab_size)

        # Get node vocab mapping
        self.node_to_index, self.index_to_node, self.nodes_vocab_size = \
            Common.load_vocab_from_dict(self.node_to_count, add_values=[Common.PAD, Common.UNK], max_size=None)
        print('Loaded nodes vocab. size: %d' % self.nodes_vocab_size)


class Code2SeqDataset(Dataset):
    def __init__(self, data_file, config, dictionaries):
        super(Dataset, self).__init__()
        data_file += ".h5"
        self.data_file = h5py.File((config.H5_FOLDER / data_file), mode='r')
        self.config = config

        self.dictionaries = dictionaries

        self.max_length_target = self.config.MAX_TARGET_PARTS
        self.max_length_leaf = self.config.MAX_NAME_PARTS
        self.max_length_ast_path = self.config.MAX_PATH_LENGTH

    def get_loaders(self, test_size=0.1, val_size=0.2, batch_size=None, shuffle=True, seed=None):
        indices = list(range(len(self)))
        if not batch_size:
            batch_size = self.config.BATCH_SIZE
        if shuffle:
            if seed:
                np.random.seed(seed)
            np.random.shuffle(indices)
        val_start, test_start = int(1 - val_size - test_size), int(1 - test_size)
        train_indices, val_indices, test_indices = indices[: val_start], \
                                                   indices[val_start: test_start], \
                                                   indices[test_start:]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = torch.utils.data.DataLoader(self, batch_size=batch_size,
                                                   sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(self, batch_size=batch_size,
                                                        sampler=valid_sampler)
        test_loader = torch.utils.data.DataLoader(self, batch_size=batch_size,
                                                        sampler=test_sampler)

        return train_loader, validation_loader, test_loader

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
        target_vector = torch.zeros(size=(self.max_length_target+1,))

        start_leaf_mask = torch.zeros(size=(self.config.MAX_CONTEXTS, self.max_length_leaf))
        end_leaf_mask = torch.zeros(size=(self.config.MAX_CONTEXTS, self.max_length_leaf))
        target_mask = torch.zeros(size=(self.max_length_target,))

        context_mask = torch.zeros(size=(self.config.MAX_CONTEXTS,))
        ast_path_lengths = torch.zeros(size=(self.config.MAX_CONTEXTS,))

        for i, context in enumerate(contexts):
            leaf_node_1, ast_path, leaf_node_2 = context.split(',')

            leaf_node_1 = leaf_node_1[:self.max_length_leaf]
            ast_path = ast_path[:self.max_length_ast_path]
            leaf_node_2 = leaf_node_2[:self.max_length_leaf]

            ast_path = [self.dictionaries.node_to_index.get(
                w, self.dictionaries.node_to_index[Common.UNK]
            ) for w in ast_path.split('|')]

            leaf_node_1 = [self.dictionaries.subtoken_to_index.get(
                w, self.dictionaries.subtoken_to_index[Common.UNK]
            ) for w in leaf_node_1.split('|')]

            leaf_node_2 = [self.dictionaries.subtoken_to_index.get(
                w, self.dictionaries.subtoken_to_index[Common.UNK]
            ) for w in leaf_node_2.split('|')]

            start_leaf_matrix[i, :len(leaf_node_1)] = torch.tensor(leaf_node_1)
            start_leaf_mask[i, :len(leaf_node_1)] = torch.ones(size=(len(leaf_node_1),))

            ast_path_matrix[i, :len(ast_path)] = torch.tensor(ast_path)
            ast_path_lengths[i] = torch.tensor(len(ast_path))

            end_leaf_matrix[i, :len(leaf_node_2)] = torch.tensor(leaf_node_2)
            end_leaf_mask[i, :len(leaf_node_2)] = torch.ones(size=(len(leaf_node_2),))

        context_mask[:len(contexts)] = torch.ones(size=(len(contexts),))

        word = word[:self.max_length_target]
        target = [self.dictionaries.target_to_index.get(
            w, self.dictionaries.target_to_index[Common.UNK]
        ) for w in word.split('|')]
        target_vector[0] = torch.tensor(self.dictionaries.target_to_index[Common.SOS])
        target_vector[1:len(target) + 1] = torch.tensor(target)
        target_mask[:len(target)] = torch.ones(size=(len(target),))

        return (start_leaf_matrix, ast_path_matrix, end_leaf_matrix, target_vector,
                start_leaf_mask, end_leaf_mask, target_mask, context_mask,
                ast_path_lengths)


def get_loaders(config, dictionaries):
    test_set = Code2SeqDataset('test', config=config, dictionaries=dictionaries)
    train_set = Code2SeqDataset('train', config=config, dictionaries=dictionaries)
    val_set = Code2SeqDataset('val', config=config, dictionaries=dictionaries)

    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    return {
        'TRAIN_LOADER': train_loader,
        'VAL_LOADER': val_loader,
        'TEST_LOADER': test_loader
    }

if __name__ == "__main__":
    config = Config.get_default_config(None)

    dictionaries = Dictionaries(config)
    loaders = get_loaders(config, dictionaries)

    for i, train_data in enumerate(loaders['TRAIN_LOADER']):
        print("start_leaf_matrix", train_data[0].shape)
        print("ast_path_matrix", train_data[1].shape)
        print("end_leaf_matrix", train_data[2].shape)
        print("target_vector", train_data[3].shape)
        print("start_leaf_mask", train_data[4].shape)
        print("end_leaf_mask", train_data[5].shape)
        print("target_mask", train_data[6].shape)
        print("context_mask", train_data[7].shape)
        print("ast_path_lengths", train_data[8].shape)

        for i in range(train_data[3].shape[0]):
            print(" ".join([dictionaries.index_to_target.get(x.item(), Common.UNK) for x in train_data[3][i]]))

        break
