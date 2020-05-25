from common import Common
from config import Config
import pickle
from random import shuffle
import h5py
import torch
from torch.utils.data import DataLoader, Dataset

torch.backends.cudnn.deterministic = True
torch.manual_seed(999)

class Dictionaries:
    def __init__(self, conf):
        self.config = conf
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
    def __init__(self, data_file, conf, dicts):
        super(Dataset, self).__init__()
        data_file += ".h5"
        self.data_file = h5py.File((conf.H5_FOLDER / data_file), mode='r')
        self.config = conf

        self.dictionaries = dicts

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
        target_vector = torch.zeros(size=(self.max_length_target + 1,))

        start_leaf_mask = torch.zeros(size=(self.config.MAX_CONTEXTS, self.max_length_leaf))
        end_leaf_mask = torch.zeros(size=(self.config.MAX_CONTEXTS, self.max_length_leaf))
        target_mask = torch.zeros(size=(self.max_length_target,))

        context_mask = torch.zeros(size=(self.config.MAX_CONTEXTS,))
        ast_path_lengths = torch.zeros(size=(self.config.MAX_CONTEXTS,))

        for idx, context in enumerate(contexts):
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

            start_leaf_matrix[idx, :len(leaf_node_1)] = torch.tensor(leaf_node_1)
            start_leaf_mask[idx, :len(leaf_node_1)] = torch.ones(size=(len(leaf_node_1),))

            ast_path_matrix[idx, :len(ast_path)] = torch.tensor(ast_path)
            ast_path_lengths[idx] = torch.tensor(len(ast_path))

            end_leaf_matrix[idx, :len(leaf_node_2)] = torch.tensor(leaf_node_2)
            end_leaf_mask[idx, :len(leaf_node_2)] = torch.ones(size=(len(leaf_node_2),))

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


def get_loaders(conf, dicts):
    torch.seed()
    test_set = Code2SeqDataset('test', conf=conf, dicts=dicts)
    train_set = Code2SeqDataset('train', conf=conf, dicts=dicts)
    val_set = Code2SeqDataset('val', conf=conf, dicts=dicts)

    test_loader = DataLoader(test_set, batch_size=conf.BATCH_SIZE, shuffle=True, num_workers=conf.NUM_WORKERS)
    train_loader = DataLoader(train_set, batch_size=conf.BATCH_SIZE, shuffle=True, num_workers=conf.NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=conf.BATCH_SIZE, shuffle=True, num_workers=conf.NUM_WORKERS)

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

        for j in range(train_data[3].shape[0]):
            print(" ".join([dictionaries.index_to_target.get(x.item(), Common.UNK) for x in train_data[3][j]]))

        break
