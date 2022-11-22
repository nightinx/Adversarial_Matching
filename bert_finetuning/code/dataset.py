import jsonlines
import re
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import random
from transformers import BertTokenizer, BertModel
def get_data(read_path1,read_path2):
    dict_hypo_theoryset={}
    
    pattern = r'sent\d+: '
    with jsonlines.open(read_path1, "r") as rfd:
        for data in rfd:
            result = re.split(pattern, data['context'])
            result_n = set([x.strip() for x in result if x.strip()!=''])
            dict_hypo_theoryset[data['hypothesis']]=result_n
    rfd.close()
    
    theory=[]
    pos_hypo=[]
    for hypo in dict_hypo_theoryset.keys():
        para=""
        for sent in dict_hypo_theoryset[hypo]:
            para+=sent+'. '
        theory.append(para.strip())
        pos_hypo.append(hypo)
        
    neg_hypo=[]
    
    with jsonlines.open(read_path2, "r") as rfd:
        for data in rfd:
            for key in data.keys():
                neg_hypo.append(data[key].split('.'))
    rfd.close()        
    
    def get_data(pos,neg,theory):
        data={'theory':[],'neg':[],'pos':[]}
        for i,t in enumerate(theory):
            data['theory'].append(t)
            data['neg'].append(neg[i])
            data['pos'].append(pos[i])
        return data
    
    data=get_data(pos_hypo,neg_hypo,theory)
    return data


class InputDataset(Dataset):
    def __init__(self, data,tokenizer,sent_len,data_size,split=0.8,mode='train'):
        self.data=data
        self.sent_len=sent_len
        self.data_size=data_size
        self.tokenizer=tokenizer
        self.split=split
        self.mode=mode
        
    def __len__(self,):
        return self.data_size
    
    def __getitem__(self,item):
        x=np.random.rand(1)*100
        assert self.mode in ['train','test'],"mode must be train or test"
        if self.mode=='train':
            item=item%(int(len(self.data['theory'])*self.split))
        elif self.mode=='test':
            item=item%(int(len(self.data['theory'])*self.split))+int((1-self.split)*len(self.data['theory']))
        
            
        if x[0]<=25:
            label=1
            hypo=self.data['pos'][item][0]
        else:
            label=0
            hypo=random.choice(self.data['neg'][item])
        label=torch.tensor(label,dtype=torch.long)
        theory=self.data['theory'][item]
        
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
    
train_data_size=100000
test_data_size=20000
read_path1='entailment_trees_emnlp2021_data_v3/dataset/task_1/train.jsonl'
read_path2='fullresult.jsonlines'

if __name__ == '__main__':
    
    data=get_data(read_path1,read_path2)
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset=InputDataset(data=data,tokenizer=tokenizer,sent_len= 500,data_size= train_data_size,split=0.8,mode='train')
    test_dataset=InputDataset(data=data,tokenizer=tokenizer,sent_len= 500,data_size= test_data_size,split=0.2,mode='test')
    train_data_loader=DataLoader(train_dataset,batch_size=4)
    test_data_loader=DataLoader(test_dataset,batch_size=1)




    batch = next(iter(train_data_loader))

    print(batch)
    print(batch['input_ids'].shape)
    print(batch['attention_mask'].shape)
    print(batch['token_type_ids'].shape)
    print(batch['labels'].shape)

    