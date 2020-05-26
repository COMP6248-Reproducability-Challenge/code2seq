import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm

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

            state={
		'epoch': epoch,
    		'model': model,
    		'optimizer': optimizer
            }
            #load the model
            state=torch.load('checkpoint_epoch_0.tar')

	    #to fix
            epoch.load_state_dict(state['epoch'])
            model.load_state_dict(state['model'])	
            optimizer.load_state_dict(state['optimizer'])
           


            for i, batch in enumerate(train_loader):
            
               print("loop is:  ", i)
               start_leaf, ast_path, end_leaf, target, start_leaf_mask, end_leaf_mask, target_mask, context_mask, ast_path_lengths = batch
        
               pred = model(*batch)
               loss = masked_cross_entropy(pred.contiguous(), 
                                            target.contiguous())
               #print("loss is : ", loss)
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()

               epoch_loss = (epoch_loss*i + loss.item()) / (i+1)
               t.set_postfix(loss='{:05.3f}'.format(epoch_loss))
               t.update()

               #remove this to train properly
               if(i==1):
                   break

	#save the model
            filename="data/checkpoint_epoch_%s.tar" %epoch
            torch.save(state, filename)


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

    train(model, optimizer, loaders, epochs=1)

	
