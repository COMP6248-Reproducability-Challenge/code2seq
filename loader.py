import os
import pickle
from linecache import getline
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import Common
from config import Config


class Code2SeqDataset(Dataset):
    def __init__(self, data_file, config, max_dict):
        self.data_file = None
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

        print("Processing...")
        self.data_file = data_file
        self.line_observations = []
        # max_length_start_leaf, max_length_ast_path, max_length_end_leaf, max_length_target = 0, 0, 0, 0
        with open(self.data_file, 'r') as data:
            num_lines = sum(1 for line in data)
        with open(self.data_file, 'r') as data:
            pbar = tqdm(total=num_lines)
            line = 0
            # for _ in range(10_000):
            while data:
                if line > num_lines:
                    break
                row = data.readline().split()
                if len(row) == 0:
                    line += 1
                    self.line_observations.append(0)
                    continue
                # word = row[0]
                # contexts = row[1:self.config.MAX_CONTEXTS + 1]
                # len_target = len(word.split("|"))
                # for context in contexts:
                #     leaf_node_1, ast_path, leaf_node_2 = context.split(',')
                #     len_start_leaf = len(leaf_node_1.split("|"))
                #     len_ast = len(ast_path.split("|"))
                #     len_end_leaf = len(leaf_node_2.split("|"))
                #     if len_start_leaf > max_length_start_leaf:
                #         max_length_start_leaf = len_start_leaf
                #     if max_length_end_leaf < len_end_leaf:
                #         max_length_end_leaf = len_end_leaf
                #     if len_ast > max_length_ast_path:
                #         max_length_ast_path = len_ast
                # if len_target > max_length_target:
                #     max_length_target = len_target
                # self.line_observations.append(len(contexts))
                line += 1
                pbar.update(1)
                self.line_observations.append(min(len(row)-1, config.MAX_CONTEXTS))
            pbar.close()
        # self.max_length_target = max_length_target
        # self.max_length_start_leaf = max_length_start_leaf
        # self.max_length_ast_path = max_length_ast_path
        # self.max_length_end_leaf = max_length_end_leaf
        self.max_length_target = max_dict['max_length_target']
        self.max_length_start_leaf = max_dict['max_length_start_leaf']
        self.max_length_ast_path = max_dict['max_length_ast_path']
        self.max_length_end_leaf = max_dict['max_length_end_leaf']

    def __len__(self):
        return sum(self.line_observations)

    def __getitem__(self, item):
        running_total = 0
        for i, x in enumerate(self.line_observations):
            if running_total + x >= item:
                line = i + 1
                break
            running_total += x

        row = getline(os.path.abspath(self.data_file), line)
        row = row.split()
        word = row[0]
        target = [self.target_to_index.get(w, self.target_to_index['<UNK>']) for w in word.split('|')]
        context = row[item - running_total]
        leaf_node_1, ast_path, leaf_node_2 = context.split(',')
        leaf_node_1 = [self.subtoken_to_index.get(w, self.subtoken_to_index['<UNK>']) for w in leaf_node_1.split('|')]
        ast_path = [self.node_to_index.get(w, self.node_to_index['<UNK>']) for w in ast_path.split('|')]
        leaf_node_2 = [self.subtoken_to_index.get(w, self.subtoken_to_index['<UNK>']) for w in leaf_node_2.split('|')]

        start_leaf_vector = torch.zeros(size=(self.max_length_start_leaf,))
        start_leaf_vector[:len(leaf_node_1)] = torch.tensor(leaf_node_1)

        ast_path_vector = torch.zeros(size=(self.max_length_ast_path,))
        ast_path_vector[:len(ast_path)] = torch.tensor(ast_path)

        end_leaf_vector = torch.zeros(size=(self.max_length_end_leaf,))
        end_leaf_vector[:len(leaf_node_2)] = torch.tensor(leaf_node_2)

        target_vector = torch.zeros(size=(self.max_length_target,))
        target_vector[:len(target)] = torch.tensor(target)

        return start_leaf_vector, ast_path_vector, end_leaf_vector, target_vector


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--load_path', '-l', dest='load_path', default=None, help="Load path for dataset.")
    parser.add_argument('--save_path', '-s', dest='save_path', default=None, help="Load path for dataset.")
    args = parser.parse_args()

    config = Config.get_default_config(None)

    # Was easier to just hard code the maximum length for each variable
    max_dict = {'max_length_target': 40,
                'max_length_start_leaf': 140,
                'max_length_ast_path': 10,
                'max_length_end_leaf': 205}

    test_set = Code2SeqDataset(config.TEST_PATH, config=config, max_dict=max_dict)
    train_set = Code2SeqDataset(config.TRAIN_PATH, config=config, max_dict=max_dict)
    val_set = Code2SeqDataset(config.VAL_PATH, config=config, max_dict=max_dict)

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    for (start_leaf, ast_path, end_leaf, target) in train_loader:
        print('Leaf 1', start_leaf)
        print('AST Path', ast_path)
        print('Leaf 2', end_leaf)
        print('Target', target)
        print(" ".join([test_set.index_to_target.get(x.item(), '??') for x in target[0]]))

