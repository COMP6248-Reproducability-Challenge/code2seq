import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm

import time

from model import Code2Seq, config
from loader import Dictionaries, get_loaders


_mce = nn.CrossEntropyLoss(size_average=True, ignore_index=0)
def masked_cross_entropy(logits, target):
    return _mce(logits.view(-1, logits.size(-1)), target.view(-1))


def train(model, optimizer, loaders, epochs=1):
    train_loader = loaders['TRAIN_LOADER']

    for epoch in range(epochs):
        with tqdm(total=len(train_loader), desc='TRAIN') as t:
            epoch_loss = 0.0
            for i, batch in enumerate(train_loader):
               start_leaf, ast_path, end_leaf, target, start_leaf_mask, end_leaf_mask, target_mask, context_mask, ast_path_lengths = batch
        
               pred = model(*batch)
               loss = masked_cross_entropy(pred.contiguous(), 
                                            target.contiguous())
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()

               epoch_loss = (epoch_loss*i + loss.item()) / (i+1)
               t.set_postfix(loss='{:05.3f}'.format(epoch_loss))
               t.update()

               if i % 200 == 0:
                   ms = int(round(time.time()*1000))
                   file_ = 'data/checkpoint_epoch_{}_{}_{}.tar'.format(i, epoch, ms)
                   torch.save(model, file_)
                   print('Model saved')


if __name__=='__main__':
    seed = 7
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dicts = Dictionaries(config)
    loaders = get_loaders(config, dicts, device)

    model = Code2Seq(dicts).to(device)
    model.train(True)

    optimizer = optim.Adam(model.parameters())

    train(model, optimizer, loaders, epochs=10)
