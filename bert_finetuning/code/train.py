

import torch
import torch.nn as nn
from tqdm import tqdm


def loss_fn(output, targets):
    return nn.BCEWithLogitsLoss()(output, targets.view(-1,1))

def train_func(data_loader, model, optimizer, device, scheduler):
    model.to(device)
    model.train()
    
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["input_ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["attention_mask"]
        targets = d["label"]
        
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        output = model(
            input_ids=ids,
            attention_mask = mask,
            token_type_ids = token_type_ids
        )
        
        
        loss = loss_fn(output, targets)
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
def eval_func(data_loader, model, device):
    model.eval()
    
    fin_targets = []
    fin_output = []
    
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["input_ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["attention_mask"]
            targets = d["label"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)


            output = model(
                input_ids=ids,
                attention_mask = mask,
                token_type_ids = token_type_ids
            )
        
            fin_targets.extend(targets.cpu().detach().numpy().to_list())
            fin_targets.extend(torch.sigmoid(output).cpu().detach().numpy().to_list())
            
        return fin_output, fin_targets
    