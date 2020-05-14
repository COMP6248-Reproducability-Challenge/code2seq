import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader

from common import Common
from config import Config


class Loader:
    def __init__(self, data_path, config, dataset_load_path=None, dataset_save_path=None):
        if dataset_load_path:
            with open(dataset_load_path, 'rb') as f:
                self.dataset = pickle.load(f)
        else:
            with open(config.DICT_PATH, 'rb') as f:
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
                                                max_size=config.SUBTOKENS_VOCAB_MAX_SIZE)
                print('Loaded subtoken vocab. size: %d' % self.subtoken_vocab_size)

                # Get target vocab mapping
                self.target_to_index, self.index_to_target, self.target_vocab_size = \
                    Common.load_vocab_from_dict(self.target_to_count, add_values=['<PAD>', '<UNK>', '<S>'],
                                                max_size=config.TARGET_VOCAB_MAX_SIZE)
                print('Loaded target word vocab. size: %d' % self.target_vocab_size)

                # Get node vocab mapping
                self.node_to_index, self.index_to_node, self.nodes_vocab_size = \
                    Common.load_vocab_from_dict(self.node_to_count, add_values=['<PAD>', '<UNK>'], max_size=None)
                print('Loaded nodes vocab. size: %d' % self.nodes_vocab_size)

                self.dataset = self.getCode2SeqDataset(data_path, self.subtoken_to_index,
                                          self.target_to_index, self.node_to_index,
                                          config)
        if dataset_save_path:
            with open(dataset_save_path, 'wb') as f:
                pickle.dump(self.dataset, f)

        self.data_loader = DataLoader(self.dataset, batch_size=config.BATCH_SIZE)

    def getCode2SeqDataset(self, data_file, subtoken2idx, target2idx, node2idx, config):
        print("Processing Code2Seq dataset...")
        seq_start_leaves, seq_ast_paths, seq_end_leaves, seq_target = [], [], [], []
        max_length_start_leaf, max_length_ast_path, max_length_end_leaf, max_length_target = 0, 0, 0, 0
        with open(data_file, 'r') as data:
            num_lines = sum(1 for line in data)
        with open(data_file, 'r') as data:
            print(num_lines)
            line = 0
            while data:
            # for _ in range(10_000):
                if line % 9999 == 0:
                    print("%s%% complete." % (line*100 // num_lines))
                row = data.readline().split()
                if len(row) == 0:
                    line += 1
                    continue
                word = row[0]
                target = [target2idx.get(w, '<UNK>') for w in word.split('|')]

                # Maay have to shuffle contexts
                contexts = row[1:config.MAX_CONTEXTS + 1]

                for context in contexts:
                    leaf_node_1, ast_path, leaf_node_2 = context.split(',')
                    leaf_node_1 = [subtoken2idx.get(w, '<UNK>') for w in leaf_node_1.split('|')]
                    ast_path = [node2idx.get(w, '<UNK>') for w in ast_path.split('|')]
                    leaf_node_2 = [subtoken2idx.get(w, '<UNK>') for w in leaf_node_2.split('|')]

                    len_start_leaf = len(leaf_node_1)
                    if len_start_leaf > max_length_start_leaf:
                        max_length_start_leaf = len_start_leaf

                    len_end_leaf = len(leaf_node_2)
                    if len_end_leaf > max_length_end_leaf:
                        max_length_end_leaf = len_end_leaf

                    len_ast = len(ast_path)
                    if len_ast > max_length_ast_path:
                        max_length_ast_path = len_ast

                    seq_start_leaves += [leaf_node_1]
                    seq_ast_paths += [ast_path]
                    seq_end_leaves += [leaf_node_2]
                    seq_target += [target]

                len_target = len(target)
                if len_target > max_length_target:
                    max_length_target = len_target

                line += 1

        # Create target matrix
        target_matrix = torch.zeros(size=(len(seq_target), max_length_target))
        for i, target in enumerate(seq_target):
            target_matrix[i, 0:len(target)] = torch.tensor(target)

        # Create start leaf matrix
        start_leaf_matrix = torch.zeros(size=(len(seq_start_leaves), max_length_start_leaf))
        for i, leaf in enumerate(seq_start_leaves):
            start_leaf_matrix[i, 0:len(leaf)] = torch.tensor(leaf)

        # Create end leaf matrix
        end_leaf_matrix = torch.zeros(size=(len(seq_end_leaves), max_length_end_leaf))
        for i, leaf in enumerate(seq_end_leaves):
            end_leaf_matrix[i, 0:len(leaf)] = torch.tensor(leaf)

        # Create ast path matrix
        ast_path_matrix = torch.zeros(size=(len(seq_ast_paths), max_length_ast_path))
        for i, path in enumerate(seq_ast_paths):
            ast_path_matrix[i, 0:len(path)] = torch.tensor(path)

        return TensorDataset(start_leaf_matrix, ast_path_matrix, end_leaf_matrix, target_matrix)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--load_path', '-l', dest='load_path', default=None, help="Load path for dataset.")
    parser.add_argument('--save_path', '-s', dest='save_path', default=None, help="Load path for dataset.")
    args = parser.parse_args()

    config = Config.get_default_config(None)

    l = Loader(data_path=config.TRAIN_PATH, dataset_load_path=args.load_path, dataset_save_path=args.save_path, config=config)

    train_loader = l.data_loader

    for (leaf1, ast, leaf2, target) in train_loader:
        print('Leaf 1', leaf1)
        print('AST Path', ast)
        print('Leaf 2', leaf2)
        print('Target', target)
        break

    print(" ".join([l.index_to_target.get(x.item(), '??') for x in target[0]]))
