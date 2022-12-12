import time
import numpy as np
from torch import nn
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
from dataset import get_data,InputDataset
import argparse
import os
from utils import set_seed
def add_learner_params():
    parser=argparse.ArgumentParser()
    # trainer params
    parser.add_argument('--train_data_size', default=200000, type=int, help='length of training data size')
    parser.add_argument('--test_data_size', default=20000, type=int, help='length of test data size')
    parser.add_argument('--read_path1', default='./data/entailment_trees_emnlp2021_data_v3/dataset/task_1/train.jsonl', 
    type=str, help='read from entailment dataset')
    parser.add_argument('-read_path2', default='./data/aligened_tree/aligened_tree.jsonlines', type=str, help='read from aligned')
    parser.add_argument('--sent_len', default=500, type=int, help='max length of sentences fed into bert')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size for both training and eval')
    parser.add_argument('--epochs', default=100, type=int, help='epoch for training')
    parser.add_argument('--save_dir', default='./bert_finetuning', type=str, help='save_path')
    args=parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    return args


def train(args):

    batch_size=args.batch_size
    EPOCHS=args.epochs
    device=args.device
    #data=get_data(read_path1,read_path2)
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset=InputDataset(read_path1=args.read_path1,read_path2=args.read_path2,tokenizer=tokenizer,sent_len= args.sent_len,data_size= args.train_data_size,split=0.8,mode='train')
    test_dataset=InputDataset(read_path1=args.read_path1,read_path2=args.read_path2,tokenizer=tokenizer,sent_len= args.sent_len,data_size= args.test_data_size,split=0.2,mode='test')
    model = BertForSeq.from_pretrained('bert-base-uncased')
    
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size)
    val_dataloader = DataLoader(test_dataset,batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * EPOCHS  # len(dataset)*epochs / batchsize
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    total_t0 = time.time()

    log = log_creater(output_dir=os.path.join(args.save_dir,'cache/logs/'))
    log.info("   args = {}".format(args))
    log.info("   Train batch size = {}".format(batch_size))
    log.info("   Total steps = {}".format(total_steps))
    log.info("   Training Start!")

    for epoch in range(EPOCHS):
        total_train_loss = 0
        t0 = time.time()
        model.to(device)
        model.train()
        for step, batch in enumerate(train_dataloader):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            model.zero_grad()

            outputs = model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels)

            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_time = format_time(time.time() - t0)

        log.info('====Epoch:[{}/{}] avg_train_loss={:.5f}===='.format(epoch+1,EPOCHS,avg_train_loss))
        log.info('====Training epoch took: {:}===='.format(train_time))
        log.info('Running Validation...')

        model.eval()
        avg_val_loss, avg_val_acc = evaluate(model, val_dataloader,args)
        val_time = format_time(time.time() - t0)
        log.info('====Epoch:[{}/{}] avg_val_loss={:.5f} avg_val_acc={:.5f}===='.format(epoch+1,EPOCHS,avg_val_loss,avg_val_acc))
        log.info('====Validation epoch took: {:}===='.format(val_time))
        log.info('')

        if epoch == EPOCHS-1:
            savename=f"{time.strftime('%Y-%m-%d-%H-%M')}"
            torch.save(model.state_dict(),os.path.join(args.save_dir,f'cache/model_{savename}.bin'))
            print('Model Saved!')
    log.info('')
    log.info('   Training Completed!')
    print('Total training took{:} (h:mm:ss)'.format(format_time(time.time() - total_t0)))

def evaluate(model,val_dataloader,args):
    total_val_loss = 0
    corrects = []
    device=args.device
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels)

        logits = torch.argmax(outputs.logits,dim=1)
        ## put accuracy per batch into a list
        ## gpu-> cpu
        preds = logits.detach().cpu().numpy()
        labels_ids = labels.to('cpu').numpy()
        # print(preds)
        # print(labels_ids)
        corrects.append((preds == labels_ids).mean())  ## [0.8,0.7,0.9]
        ## get loss
        loss = outputs.loss
        ## loss per batch -> total_val_loss
        ## len(val_dataloader) batches in total
        total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_acc = np.mean(corrects)

    return avg_val_loss, avg_val_acc

def log_creater(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir, log_name)
    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] ==> %(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))
    return log

if __name__ == '__main__':
    set_seed()
    args=add_learner_params()
    train(args)
