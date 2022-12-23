from sentence_transformers import SentenceTransformer
import sys
from ebdataset import EBdataset
import numpy as np
import torch
import os
from utils import *
import os
import torch
import logging
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from transformers.utils.notebook import format_time
from model import *
import argparse
def add_learner_params():
    parser=argparse.ArgumentParser()
    # trainer params
    parser.add_argument('--sent_len', default=500, type=int, help='max length of sentences fed into bert')
    parser.add_argument('--batch_size', default=460, type=int, help='batch size for both training and eval')
    parser.add_argument('--model_path', default='./bert_finetuning/cache/saved_model_16_50000_5000/model_2022-12-17-05-03.bin', type=str, help='model path')
    args=parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.device)
    return args

def load_model(args):
    model = BertForSeq.from_pretrained('bert-base-uncased')
    device=args.device
    model.load_state_dict(torch.load(args.model_path))
    return model.to(device)

def eval_batch(args,model,batch):
    device=args.device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    token_type_ids = batch['token_type_ids'].to(device)
    labels = batch['labels'].to(device)

    with torch.no_grad():
        outputs = model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels)
    return torch.softmax(outputs.logits,dim=1)[:,1]

class Eval_Dataset(Dataset):
    def __init__(self, data,tokenizer,args):
        self.data=data
        self.sent_len=args.sent_len
        self.tokenizer=tokenizer
    
        
    def __len__(self,):
        return len(self.data)*len(self.data)
    
    def __getitem__(self,item):        
        label=torch.tensor(1,dtype=torch.long)
        theory=self.data[int(item/len(self.data))].topara('golden')
        hypo=self.data[int(item/len(self.data))].neg[item%len(self.data)]
        encoding=self.tokenizer.encode_plus(
            theory,
            hypo,
            add_special_tokens=True, #add [CLS] and [SEP]
            max_length=self.sent_len,#max input length
            return_token_type_ids=True,#theory 11111 and hypo 00000
            pad_to_max_length=True,# fill or cut up to max input length 
            return_attention_mask=True,# attention encoding
            return_tensors='pt'# pytorch model
        )
        
        return {
            "theory":theory,
            "hypo":hypo,
            "input_ids":encoding['input_ids'].flatten(),
            "attention_mask":encoding['attention_mask'].flatten(),
            "token_type_ids":encoding['token_type_ids'].flatten(),
            "labels":label
        }

def relevance(tsk:EBdataset):
    args=add_learner_params()
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    dataset=Eval_Dataset(tsk.data,tokenizer,args)
    data_len=len(tsk.data)
    dataloader=DataLoader(dataset,batch_size=args.batch_size)
    model=load_model(args)

    save_mat=torch.zeros((data_len,data_len)).to(args.device)
    row_idx=0
    col_idx=0

    for idx,batch in enumerate(dataloader):
        print(idx)
        logits=eval_batch(args,model,batch)
        save_mat[row_idx][col_idx:col_idx+args.batch_size]=logits
        col_idx+=args.batch_size
        if col_idx>=data_len:
            col_idx=0
            row_idx+=1
    save_mat=save_mat.detach().cpu().numpy()
    print(save_mat)
    return save_mat


if __name__=='__main__':
    import sys
    from ebdataset import EBdataset
    from alignment import *
    from alignment import align_response,get_doc,write_np
    from utils import *
    tsk1_path='/home/cc/github/Adversarial_Matching/data/entailment_trees_emnlp2021_data_v3/dataset/task_1'
    tsk2_path='/home/cc/github/Adversarial_Matching/data/entailment_trees_emnlp2021_data_v3/dataset/task_2'
    tsk1=EBdataset(tsk1_path,tsk2_path)
    tsk1.get_neg(osp.join('data/alignment','neg.npy'))
    tsk1.get_tsk1_table()
    b=relevance(tsk1)
    savenp('./data/relevance','test.npy',b)