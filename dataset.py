# MIT License
#
# Copyright (c) 2019 hukuda222
#
# https://github.com/hukuda222/code2seq
from torch.utils.data import Dataset
import torch
from numpy import random as rnd


class C2SDataSet(Dataset):
    def __init__(self, config, filedata, data_size, terminal_dict, path_dict,
                 target_dict, device):
        super(Dataset, self).__init__()
        self.f = filedata
        self.size = data_size
        self.target_dict = target_dict
        self.path_dict = path_dict
        self.terminal_dict = terminal_dict
        self.device = device
        self.max_context_length = config.MAX_CONTEXTS
        self.max_terminal_length = config.MAX_NAME_PARTS
        self.max_path_length = config.MAX_PATH_LENGTH
        self.max_target_length = config.MAX_TARGET_PARTS

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        line = self.f[str(index)]["row"][()]
        ss = line.split(" ")
        sss = [s.split(",") for s in ss[1:] if len(s) > 3]
        starts = []
        paths = []
        ends = []
        context_mask = []
        target = []
        start_mask = []
        end_mask = []
        path_length = []
        target_mask = []

        sss_shuffled_index = [i for i in range(len(sss))]
        rnd.shuffle(sss_shuffled_index)
        for sss_i in sss_shuffled_index[:self.max_context_length]:
            s = sss[sss_i]
            if len(s) != 3:
                continue
            # ここで一つ目の末端文字列
            start = []
            for ter1_s in s[0].split("|")[:self.max_terminal_length]:
                start.append(self.terminal_dict[ter1_s] if ter1_s in
                             self.terminal_dict else
                             self.terminal_dict["<UNK>"])
            start_mask.append([1] * len(start) + [0] *
                              (self.max_terminal_length - len(start)))
            start += [self.terminal_dict["<PAD>"]] * \
                (self.max_terminal_length - len(start))
            starts.append(start)

            # ここでpath
            path = []
            for path_e in s[1].split("|")[:self.max_path_length]:
                path.append(
                    self.path_dict[path_e] if path_e in
                    self.path_dict else self.path_dict["<UNK>"])
            path_length.append(len(path))
            path += [self.path_dict["<PAD>"]] * \
                (self.max_path_length - len(path))

            paths.append(path)

            # ここで二つ目の末端文字列
            end = []
            for ter2_s in s[2].split("|")[:self.max_terminal_length]:
                end.append(self.terminal_dict[ter2_s] if ter2_s in
                           self.terminal_dict else
                           self.terminal_dict["<UNK>"])
            end_mask.append([1] * len(end) + [0] *
                            (self.max_terminal_length - len(end)))
            end += [self.terminal_dict["<PAD>"]] * \
                (self.max_terminal_length - len(end))
            ends.append(end)

            context_mask.append(1)

        pad_length = self.max_context_length - len(context_mask)
        paths += [[self.path_dict["<PAD>"]
                   for i in range(self.max_path_length)]
                  for j in range(pad_length)]
        path_length += [1] * pad_length  # 0はダメらしい(あとでattentionを0にするので多分大丈夫)
        starts += [[self.terminal_dict["<PAD>"]
                    for i in range(self.max_terminal_length)]
                   for j in range(pad_length)]
        start_mask += \
            [[0 for i in range(self.max_terminal_length)]
             for j in range(pad_length)]
        ends += [[self.terminal_dict["<PAD>"]
                  for i in range(self.max_terminal_length)]
                 for j in range(pad_length)]
        end_mask += \
            [[0 for i in range(self.max_terminal_length)]
             for j in range(pad_length)]

        context_mask += [0] * pad_length

        target.append(self.target_dict["<S>"])
        for tar_s in ss[0].split("|")[:self.max_target_length - 2]:
            target.append(self.target_dict[tar_s] if tar_s in
                          self.target_dict else
                          self.target_dict["<UNK>"])
        target.append(self.target_dict["<PAD>"])  # eos
        target_mask = [1] * (len(target)-1)  # sos
        target_mask += [0] * \
            (self.max_target_length + 1 - len(target))
        target += [self.target_dict["<PAD>"]] * \
            (self.max_target_length + 1 - len(target))

        return torch.tensor(starts, dtype=torch.long).to(self.device),\
            torch.tensor(paths, dtype=torch.long).to(self.device),\
            torch.tensor(ends, dtype=torch.long).to(self.device),\
            torch.tensor(target, dtype=torch.long).to(self.device), \
            torch.tensor(start_mask, dtype=torch.float).to(self.device), \
            torch.tensor(end_mask, dtype=torch.float).to(self.device), \
            torch.tensor(target_mask, dtype=torch.float).to(self.device), \
            torch.tensor(context_mask, dtype=torch.float).to(self.device),\
            torch.tensor(path_length, dtype=torch.int64).to(self.device)

