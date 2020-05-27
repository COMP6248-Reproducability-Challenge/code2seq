import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm

import time

from model import Code2Seq, config
from loader import Dictionaries, get_loaders
from common import Common


def train(model, optimizer, criterion, loaders, epochs=1):
    train_loader = loaders['TRAIN_LOADER']
    val_loader = loaders['VAL_LOADER']

    ms = int(round(time.time()*1000))

    for epoch in range(epochs):
        model.train(True)
        with tqdm(total=len(train_loader), desc='TRAIN') as t:
            epoch_loss = 0.0
            epoch_f1 = 0.0
            losses = []
            f1s = []
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

                epoch_loss = (epoch_loss*i + loss.item()) / (i+1)
                epoch_f1 = (epoch_f1*i + f1) / (i+1)
                t.set_postfix(loss='{:05.3f}'.format(epoch_loss),
                              f1='{:05.3f}'.format(epoch_f1),)
                t.update()

                if i % 200 == 0:
                    # Get this out to std for plotting later
                    print(epoch_loss)
                    print(epoch_f1)
                    losses.append(epoch_loss)
                    f1s.append(epoch_f1)
                    file_ = 'data/{}_iteration_{}_epoch_{}.tar'.format(ms, i, epoch)
                    torch.save(model, file_)
                    print('Model saved')

        print(losses)
        print(f1s)
        file_ = 'data/{}_epoch_{}.tar'.format(ms, epoch)
        torch.save(model, file_)
        print('Model saved')

        model.eval()
        with tqdm(total=len(val_loader), desc='VAL') as t:
            epoch_loss = 0.0
            epoch_f1 = 0.0
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

                epoch_loss = (epoch_loss*i + loss.item()) / (i+1)
                epoch_f1 = (epoch_f1*i + f1) / (i+1)
                t.set_postfix(loss='{:05.3f}'.format(epoch_loss),
                              f1='{:05.3f}'.format(epoch_f1),)
                t.update()


if __name__=='__main__':
    seed = 7
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dicts = Dictionaries(config)
    loaders = get_loaders(config, dicts, device)

    model = Code2Seq(dicts).to(device)
    model.train(True)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters())

    train(model, optimizer, criterion, loaders, epochs=10)
