import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from model import Code2Seq, config
from loader import Dictionaries, get_loaders
from common import Common


_mce = nn.CrossEntropyLoss(size_average=True, ignore_index=0)
def masked_cross_entropy(logits, target):
    return _mce(logits.view(-1, logits.size(-1)), target.view(-1))


def train(model, optimizer, loaders, epochs=10):
    train_loader = loaders['TRAIN_LOADER']
    val_loader = loaders['VAL_LOADER']

    for epoch in range(epochs):
        with tqdm(total=len(train_loader), desc='TRAIN') as t:
            epoch_loss = 0.0
            epoch_f1 = 0.0
            for i, batch in enumerate(train_loader):
                start_leaf, ast_path, end_leaf, target, start_leaf_mask, end_leaf_mask, target_mask, context_mask, ast_path_lengths = batch
                # Remove <SOS>
                target = target[:,1:]
        
                pred = model(*batch)
                loss = masked_cross_entropy(pred.contiguous(), 
                                            target.contiguous())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                target = target.contiguous().data.cpu()[:,1:]
                pred = pred.max(dim=-1)[1].data.cpu().t()

                precision, recall, f1 = Common.get_scores(*model.get_evaluation(pred, target))

                epoch_loss = (epoch_loss*i + loss.item()) / (i+1)
                epoch_f1 = (epoch_f1*i + f1) / (i+1)
                t.set_postfix(loss='{:05.3f}'.format(epoch_loss),
                              f1='{:05.3f}'.format(epoch_f1),)
                t.update()

        with tqdm(total=len(val_loader), desc='VAL') as t:
            for i, batch in enumerate(val_loader):
                start_leaf, ast_path, end_leaf, target, start_leaf_mask, end_leaf_mask, target_mask, context_mask, ast_path_lengths = batch

                pred = model(*batch)
                loss = masked_cross_entropy(pred.contiguous(),
                                            target.contiguous())

                target = target.contiguous().data.cpu()[:,1:]
                pred = pred.max(dim=-1)[1].data.cpu().t()

                precision, recall, f1 = Common.get_scores(*model.get_evaluation(pred, target))

                epoch_loss = (epoch_loss*i + loss.item()) / (i+1)
                epoch_f1 = (epoch_f1*i + f1) / (i+1)
                t.set_postfix(loss='{:05.3f}'.format(epoch_loss),
                              f1='{:05.3f}'.format(epoch_f1),)


if __name__=='__main__':
    seed = 7
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dicts = Dictionaries(config)
    loaders = get_loaders(config, dicts, device)

    model = Code2Seq(dicts).to(device)

    optimizer = optim.Adam(model.parameters())
    model.train(True)

    train(model, optimizer, loaders, epochs=10)

