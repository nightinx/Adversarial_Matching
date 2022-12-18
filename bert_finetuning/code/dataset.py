import jsonlines
import re
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import random
from transformers import BertTokenizer, BertModel
import os
def split_folds(read_path1,read_path2,num,save_path):
    data=get_data(read_path1,read_path2)
    assert len(data['theory'])%num==0
    length=int(len(data['theory'])/num)
    pos_all=np.array(data['pos'])
    neg_all=np.array(data['neg'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(num):
        test_slices=[j for j in range(i*length,(i+1)*length)]
        train_slices=[k for k in range(length*num) if k not in  test_slices]
        print(len(test_slices))
        print(len(train_slices))
        test_pos_hypo=pos_all[test_slices]
        train_pos_hypo=pos_all[train_slices]
        print(len(test_pos_hypo))
        print(len(train_pos_hypo))

        test_neg_hypo=neg_all[test_slices,:]
        test_neg_hypo=test_neg_hypo[:,test_slices]
        train_neg_hypo=neg_all[train_slices,:]
        train_neg_hypo=train_neg_hypo[:,train_slices]

        path_train=os.path.join(save_path,f'split_train_folds{i}')
        path_test=os.path.join(save_path,f'split_test_folds{i}')

        with jsonlines.open(path_train, 'a') as w:
            for index,hypo in enumerate(train_pos_hypo):
                negpara=""
                for neghypo in train_neg_hypo[index]:
                    negpara+=neghypo
                    if negpara[-1]!='.':
                        negpara+='.'
                if negpara[-1]=='.':
                    negpara=negpara[:-1]
                w.write({hypo: negpara})
        
        with jsonlines.open(path_test, 'a') as w:
            for index,hypo in enumerate(test_pos_hypo):
                negpara=""
                for neghypo in test_neg_hypo[index]:
                    negpara+=neghypo
                    if negpara[-1]!='.':
                        negpara+='.'
                if negpara[-1]=='.':
                    negpara=negpara[:-1]
                w.write({hypo: negpara})

def get_final_data(pos,neg,theory):
        data={'theory':[],'neg':[],'pos':[]}
        for i,t in enumerate(theory):
            data['theory'].append(t)
            data['neg'].append(neg[i])
            data['pos'].append(pos[i])
        return data
        
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
                neg_list=data[key].strip().split('.')
                neg_list[-1]='Chatgpt yyds'
                neg_hypo.append(neg_list)
    rfd.close()        
    return get_final_data(pos_hypo,neg_hypo,theory)

def get_data_from_folds(read_path1,folds_dir,folds_index=0,num=4):
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
    
    theory=np.array(theory)
    length=int(len(theory)/num)
    i=folds_index
    test_slices=[j for j in range(i*length,(i+1)*length)]
    train_slices=[k for k in range(length*num) if k not in test_slices]
    train_theory=theory[train_slices]
    test_theory=theory[test_slices]

    test_pos_hypo=[]
    test_neg_hypo=[]
    read_path2=os.path.join(folds_dir,f'split_test_folds{folds_index}')
    with jsonlines.open(read_path2, "r") as rfd:
        for data in rfd:
            for key in data.keys():
                neg_list=data[key].strip().split('.')
                test_neg_hypo.append(neg_list)
                test_pos_hypo.append(key)
    

    train_pos_hypo=[] 
    train_neg_hypo=[]        
    read_path2=os.path.join(folds_dir,f'split_train_folds{folds_index}')
    with jsonlines.open(read_path2, "r") as rfd:
        for data in rfd:
            for key in data.keys():
                neg_list=data[key].strip().split('.')
                train_neg_hypo.append(neg_list)
                train_pos_hypo.append(key)
    
    return (get_final_data(train_pos_hypo,train_neg_hypo,train_theory),get_final_data(test_pos_hypo,test_neg_hypo,test_theory))


class InputDataset(Dataset):
    def __init__(self, read_path1,read_path2,tokenizer,sent_len,data_size,split=0.8,mode='train'):
        self.data=get_data(read_path1,read_path2)
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
        
            
        if x[0]<=50:
            label=1
            hypo=self.data['pos'][item]
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

class InputDataset_V2(Dataset):
    def __init__(self, tokenizer,sent_len,data_size,data,random_true):
        self.data=data
        self.sent_len=sent_len
        self.data_size=data_size
        self.tokenizer=tokenizer
        self.random_true=random_true
    def __len__(self,):
        return self.data_size
    
    def __getitem__(self,item):
        x=np.random.rand(1)*100
        item%(int(len(self.data['theory'])))
        randomint=int(self.random_true*100)
            
        if x[0]<=randomint:
            label=1
            hypo=self.data['pos'][item]
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


class Eval_false_Dataset(Dataset):
    def __init__(self, read_path1,read_path2,tokenizer,sent_len,data_length):
        self.data=get_data(read_path1,read_path2)
        self.data_length=data_length
        self.data_size=self.data_length*self.data_length
        self.sent_len=sent_len
        self.tokenizer=tokenizer
        
    def __len__(self,):
        return self.data_size
    
    def __getitem__(self,item):        
        label=torch.tensor(0,dtype=torch.long)
        theory=self.data['theory'][int(item/self.data_length)]
        hypo=self.data['neg'][int(item/self.data_length)][item%self.data_length]
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

class Eval_true_Dataset(Dataset):
    def __init__(self, read_path1,read_path2,tokenizer,sent_len,data_length):
        self.data=get_data(read_path1,read_path2)
        self.data_length=data_length
        self.data_size=self.data_length
        self.sent_len=sent_len
        self.tokenizer=tokenizer
        
    def __len__(self,):
        return self.data_size
    
    def __getitem__(self,item):        
        label=torch.tensor(1,dtype=torch.long)
        theory=self.data['theory'][item]
        hypo=self.data['pos'][item]
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


if __name__ == '__main__':
    read_path1='./data/entailment_trees_emnlp2021_data_v3/dataset/task_1/train.jsonl'
    read_path2='./data/aligened_tree/aligened_tree.jsonlines'
    #split_folds(read_path1,read_path2,4,'./data/folds')
    # sys.exit()
    data=get_data(read_path1,read_path2)
    length=319
    for index in range(4):
        data_train,data_test=get_data_from_folds(read_path1,'./data/folds',index)
        for i in range(length):
            x=i+index*length
            assert data_test['pos'][i]==data['pos'][x]
            assert data_test['theory'][i]==data['theory'][x]

    for index in range(4):
        data_train,data_test=get_data_from_folds(read_path1,'./data/folds',index)
        for i in range(4*length-length):
            if i<index*lenth:
                x=i
            else:
                x=i+length
            assert data_train['pos'][i]==data['pos'][x]
            assert data_train['theory'][i]==data['theory'][x]
    
   
    
    # print(data_test['pos'])

    # for i,items in enumerate(data_train['neg']):
    #     for j,item in enumerate(items):
    #         if i<index*length:
    #             x=i
    #         else:
    #             x=i+length
    #         if item!=data['neg'][x][x]:
    #             print(i)
    #             print(item)
    #             print(data['neg'][x][x])
    #             break
    

    # tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    # # dataset=Eval_true_Dataset(read_path1=read_path1,read_path2=read_path2,tokenizer=tokenizer,sent_len= 500)
    # train_dataset=InputDataset(read_path1=read_path1,read_path2=read_path2,tokenizer=tokenizer,sent_len= 500,data_size= train_data_size,split=0.8,mode='train')
    # test_dataset=InputDataset(read_path1=read_path1,read_path2=read_path2,tokenizer=tokenizer,sent_len= 500,data_size= test_data_size,split=0.2,mode='test')
    # # data_loader=DataLoader(dataset,batch_size=1)
    # test_data_loader=DataLoader(test_dataset,batch_size=1)
    # train_data_loader=DataLoader(train_dataset,batch_size=1)



    # for step, batch in enumerate(train_data_loader):
    #     print(step)
    #     print(batch["theory"])
    #     print(batch["hypo"])
    #     print(batch["labels"])
    
    # batch = next(iter(dataset))
    # print(len(train_data_loader))
    # print(len(test_data_loader))
    # print(batch)
    # print(batch['input_ids'].shape)
    # print(batch['attention_mask'].shape)
    # print(batch['token_type_ids'].shape)
    # print(batch['labels'].shape)
    # data=get_data(read_path1,read_path2)
    # for i,item in enumerate(data['theory']):
    #     print(i,len(data['neg']))
    # for step, batch in enumerate(data_loader):
    #     print(step)
    #     if step>1000 and step<1005 :
    #         print(batch["theory"])
    #         print(batch["hypo"])
    #         print(data["theory"][step])
    #         print(data["pos"][step])
    #     if step>1005:
    #         break
    