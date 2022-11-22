from transformers import Trainer,TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel,BertConfig
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from dataset import get_data,InputDataset

train_data_size=100000
test_data_size=20000
read_path1='entailment_trees_emnlp2021_data_v3/dataset/task_1/train.jsonl'
read_path2='fullresult.jsonlines'

## model: BertForSequence
class BertForSeq(BertPreTrainedModel):

    def __init__(self,config):  ##  config.json
        super(BertForSeq,self).__init__(config)
        self.config = BertConfig(config)
        self.num_labels = 2 # num of labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask = None,
            token_type_ids = None,
            labels = None,
            return_dict = None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        )  

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,  
            logits=logits,  
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == '__main__':


    data=get_data(read_path1,read_path2)
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset=InputDataset(data=data,tokenizer=tokenizer,sent_len= 500,data_size= train_data_size,split=0.8,mode='train')
    test_dataset=InputDataset(data=data,tokenizer=tokenizer,sent_len= 500,data_size= test_data_size,split=0.2,mode='test')
    train_data_loader=DataLoader(train_dataset,batch_size=1)
    test_data_loader=DataLoader(test_dataset,batch_size=1)

    model = BertForSeq.from_pretrained('bert-base-uncased')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



    train_dataloader = DataLoader(train_dataset,batch_size=1)
    val_dataloader = DataLoader(test_dataset,batch_size=1)

    
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSeq.from_pretrained('bert-base-uncased')
    
    batch = next(iter( test_data_loader))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    token_type_ids = batch['token_type_ids'].to(device)
    labels = batch['labels'].to(device)
    model=model.to(device)
    
    model.eval()
     
    outputs = model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,labels=labels)
   
    logits = outputs.logits
    loss = outputs.loss

    print(logits)
    print(loss.item())

    preds = torch.argmax(logits,dim=1)
    print(preds)