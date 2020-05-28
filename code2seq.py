import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm
import h5py
import time
import pickle

from model import Code2Seq, config
from loader import Dictionaries, get_loaders
from common import Common
from dataset import C2SDataSet

def train(model, optimizer, criterion, train_loader, val_loader, epochs=1):
    ms = int(round(time.time() * 1000))

    out_data = []
    for epoch in range(epochs):
        model.train(True)
        with tqdm(total=len(train_loader), desc='TRAIN') as t:
            epoch_loss = 0.0
            epoch_f1 = 0.0
            train_losses = []
            train_f1s = []
            for i, batch in enumerate(train_loader):
                start_leaf, ast_path, end_leaf, target, start_leaf_mask, end_leaf_mask, target_mask, context_mask, ast_path_lengths = batch

                pred = model(*batch)
                # Remove <SOS>
                pred = pred[:, 1:]
                target = target[:, 1:]

                pred_ = pred.permute(0, 2, 1)
                # BCE loss
                result_loss = criterion(pred_, target)
                # Remove padding from loss
                loss = result_loss * target_mask

                loss = torch.sum(loss) / config.BATCH_SIZE

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                eval_ = model.get_evaluation(pred, target)
                precision, recall, f1 = Common.get_scores(*eval_)

                epoch_loss = (epoch_loss * i + loss.item()) / (i + 1)
                epoch_f1 = (epoch_f1 * i + f1) / (i + 1)
                t.set_postfix(loss='{:05.3f}'.format(epoch_loss),
                              f1='{:05.3f}'.format(epoch_f1), )
                t.update()

                if i % 200 == 0:
                    # Get this out to std for plotting later
                    print(epoch_loss)
                    print(epoch_f1)
                    train_losses.append(epoch_loss)
                    train_f1s.append(epoch_f1)
                    file_ = 'data/{}_iteration_{}_epoch_{}.tar'.format(ms, i, epoch)
                    torch.save(model, file_)
                    print('Model saved')

        print(train_losses)
        print(train_f1s)
        file_ = 'data/{}_epoch_{}.tar'.format(ms, epoch)
        torch.save(model, file_)
        print('Model saved')

        model.eval()
        with tqdm(total=len(val_loader), desc='VAL') as t:
            epoch_loss = 0.0
            epoch_f1 = 0.0
            val_losses = []
            val_f1s = []
            for i, batch in enumerate(val_loader):
                start_leaf, ast_path, end_leaf, target, start_leaf_mask, end_leaf_mask, target_mask, context_mask, ast_path_lengths = batch

                pred = model(*batch)
                # Remove <SOS>
                pred = pred[:, 1:]
                target = target[:, 1:]

                pred_ = pred.permute(0, 2, 1)
                # BCE loss
                result_loss = criterion(pred_, target)
                # Remove padding from loss
                loss = result_loss * target_mask
                loss = torch.sum(loss) / config.BATCH_SIZE

                eval_ = model.get_evaluation(pred, target)
                precision, recall, f1 = Common.get_scores(*eval_)

                epoch_loss = (epoch_loss * i + loss.item()) / (i + 1)
                epoch_f1 = (epoch_f1 * i + f1) / (i + 1)
                val_losses.append(epoch_loss)
                val_f1s.append(epoch_f1)
                t.set_postfix(loss='{:05.3f}'.format(epoch_loss),
                              f1='{:05.3f}'.format(epoch_f1), )
                t.update()
        out_data.append([train_losses, train_f1s, val_losses, val_f1s])
        with open('results_save.file', 'wb') as f:
            pickle.dump(out_data, f)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", "-e", type=int, default=1,
                        help="Number of examples in epoch")
    parser.add_argument("--datapath",
                        default="./data/java-small.dict.c2s",
                        help="path of input data")
    parser.add_argument("--trainpath",
                        default="./data/h5/train.h5",
                        help="path of train data")
    parser.add_argument("--validpath",
                        default="./data/h5/val.h5",
                        help="path of valid data")
    parser.add_argument("--trainnum", type=int, default=691974,
                        help="size of train data")
    parser.add_argument("--validnum", type=int, default=23844,
                        help="size of valid data")
    args = parser.parse_args()

    seed = 7
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dicts = Dictionaries(config)

    # model = torch.load("data/1590690332408_iteration_6800_epoch_0.tar", map_location=torch.device('cpu'))
    model = Code2Seq(dicts).to(device)
    model.train(True)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters())

    train_h5 = h5py.File(args.trainpath, 'r')
    val_h5 = h5py.File(args.validpath, 'r')

    train_set = C2SDataSet(
        config, train_h5, args.trainnum, dicts.subtoken_to_index,
        dicts.node_to_index, dicts.target_to_index, device
    )

    # train_set = Subset(train_set, list(range(1280)))

    val_set = C2SDataSet(
        config, val_h5, args.validnum, dicts.subtoken_to_index,
        dicts.node_to_index, dicts.target_to_index, device
    )

    # val_set = Subset(val_set, list(range(1280)))

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=True)


    train(model, optimizer, criterion, train_loader, val_loader, epochs=args.epoch)
