import torch
from dataset import get_data,InputDataset
from torch.utils.data import DataLoader
from model import BertForSeq
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset,DataLoader
from train import train_func,eval_func
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics

train_data_size=100000
test_data_size=20000
read_path1='entailment_trees_emnlp2021_data_v3/dataset/task_1/train.jsonl'
read_path2='fullresult.jsonlines'

def run(cfg):
    
    data=get_data(read_path1,read_path2)
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset=InputDataset(data=data,tokenizer=tokenizer,sent_len= 500,data_size= train_data_size,split=0.8,mode='train')
    test_dataset=InputDataset(data=data,tokenizer=tokenizer,sent_len= 500,data_size= test_data_size,split=0.2,mode='test')
    train_data_loader=DataLoader(train_dataset,batch_size=1)
    test_data_loader=DataLoader(test_dataset,batch_size=1)



    device = torch.device("cuda")
    model = BertModel.from_pretrained("bert-base-uncased")

    param_optimizer = list(model.named_parameters())
    no_decay = [
        "bias", 
        "LayerNorm,bias",
        "LayerNorm.weight",
               ]
    optimizer_parameters = [
        {'params': [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],
                   'weight_decay':0.001},
        {'params': [p for n,p in param_optimizer if any(nd in n for nd in no_decay)],
                   'weight_decay':0.0}
    ]

    num_train_steps = int(train_data_size/ 8*10)

    optimizers = AdamW(optimizer_parameters, lr=3e-5)

    scheduler = get_linear_schedule_with_warmup(
        optimizers,
        num_warmup_steps=0,
        num_training_steps=num_train_steps

    )

    best_accuracy = 0
    for epoch in range(5):
        train_func(data_loader=train_data_loader, model=model, optimizer=optimizers, device=device, scheduler=scheduler)
        outputs, targets = eval_func(data_loader=train_data_loader, model=model, device=device)
        outputs = np.array(outputs) >= 0.5
        val_accuracy = metrics.accuracy_score()

        outputs, targets = eval_func(data_loader=test_data_loader, model=model, device=device)
        outputs = np.array(outputs) >= 0.5
        test_accuracy = metrics.accuracy_score()
        print(f"Test Accuracy Score: {test_accuracy},Val Accuracy Score: {val_accuracy}")

        if val_accuracy>best_accuracy:
            torch.save(model.state_dict(), "model.bin")
            best_accuracy = val_accuracy
                
import json
              
if __name__ == "__main__":
    with open('config.json') as j:
        cfg = json.load(j)
    print(cfg)
    run(cfg)
    