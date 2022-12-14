import time
import numpy as np
import time
import os
import torch
import logging
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from transformers.utils.notebook import format_time
from modeling import BertForSeq
from dataset import Eval_false_Dataset_V2,Eval_true_Dataset_V2,get_data_from_folds
import argparse
import os
from utils import set_seed

#generate weight matrix of relevance
def add_learner_params():
    parser=argparse.ArgumentParser()
    # trainer params
    parser.add_argument('--read_path1', default='./data/entailment_trees_emnlp2021_data_v3/dataset/task_1/train.jsonl', 
    type=str, help='read from entailment dataset')
    parser.add_argument('-read_path2', default='./data/aligened_tree/aligened_tree.jsonlines', type=str, help='read from aligned')
    parser.add_argument('--neg_length', default=319, type=int, help='length of negtive examples')
    parser.add_argument('--sent_len', default=500, type=int, help='max length of sentences fed into bert')
    parser.add_argument('--batch_size', default=319, type=int, help='batch size for both training and eval')
    parser.add_argument('--epochs', default=1, type=int, help='epoch for training')
    parser.add_argument('--save_dir', default='./bert_finetuning', type=str, help='save_path')
    parser.add_argument('--trained_model_path', default='./bert_finetuning/cache/model_2022-12-18-06-20_1.bin', type=str, help='model path')
    parser.add_argument('--data_path', default='./data', type=str, help='data path')
    parser.add_argument('--train_fold', default=3, type=int, help='fold index')
    parser.add_argument('--folds_dir', default='./data/folds', type=str, help='location of splitted folds ')
    args=parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.device)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    return args


def test(args,model,dataloader):
    device=args.device
    save_mat=torch.zeros((args.neg_length,args.neg_length)).to(device)
    row_idx=0
    col_idx=0
    for step,batch in enumerate(dataloader):
        print(step)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels)

        logits = torch.softmax(outputs.logits,dim=1)[:,1]
        print(logits.detach().cpu().numpy())


    
def main(args):
    assert args.neg_length%args.batch_size==0

    batch_size=args.batch_size
    EPOCHS=args.epochs
    device=args.device

    model = BertForSeq.from_pretrained('bert-base-uncased')
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    model.load_state_dict(torch.load(args.trained_model_path))
    model.to(device)
    data_train,data_test=get_data_from_folds(args.read_path1,args.folds_dir,args.train_fold)
    # neg_dataset=Eval_false_Dataset(read_path1=args.read_path1,read_path2=args.read_path2,tokenizer=tokenizer,sent_len= args.sent_len,data_length=args.neg_length)
    # pos_dataset=Eval_true_Dataset(read_path1=args.read_path1,read_path2=args.read_path2,tokenizer=tokenizer,sent_len= args.sent_len,data_length=args.neg_length)
    neg_dataset=Eval_false_Dataset_V2(data=data_test,tokenizer=tokenizer,sent_len= args.sent_len,data_length=args.neg_length)
    pos_dataset=Eval_true_Dataset_V2(data=data_test,tokenizer=tokenizer,sent_len= args.sent_len,data_length=args.neg_length)

    pos_dataloader = DataLoader(pos_dataset,batch_size=batch_size)
    neg_dataloader = DataLoader(neg_dataset,batch_size=batch_size)
    test(args,model,neg_dataloader)
    

if __name__ == '__main__':
    args=add_learner_params()
    set_seed()
    main(args)
