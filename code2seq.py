import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from model import Code2Seq, config
from loader import Code2SeqDataset


_mce = nn.CrossEntropyLoss(size_average=False, ignore_index=0)
def masked_cross_entropy(logits, target):
    return _mce(logits.view(-1, logits.size(-1)), target.view(-1))


def train(model, optimizer, train_loader):
    EPOCHS = 10
    for epoch in range(EPOCHS):
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


if __name__=='__main__':
    seed = 7
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_set = Code2SeqDataset('test', config=config, device=device)
    train_set = Code2SeqDataset('train', config=config, device=device)
    val_set = Code2SeqDataset('val', config=config, device=device)

    test_loader = DataLoader(test_set, 
                             batch_size=config.BATCH_SIZE, 
                             shuffle=True, 
                             num_workers=config.NUM_WORKERS)
    train_loader = DataLoader(train_set, 
                              batch_size=config.BATCH_SIZE, 
                              shuffle=True)
    val_loader = DataLoader(val_set, 
                            batch_size=config.BATCH_SIZE, 
                            shuffle=True)


    model = Code2Seq(train_set.subtoken_vocab_size, 
                     train_set.nodes_vocab_size, 
                     train_set.target_vocab_size).to(device)

    optimizer = optim.Adam(model.parameters())
    model.train(True)

    train(model, optimizer, train_loader)

