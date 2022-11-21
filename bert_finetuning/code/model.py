from torch import nn
import numpy as np
import torch
from transformers import BertModel,BertPreTrainedModel,BertConfig
import sys

class BertForSeq(BertPreTrainedModel):
    
    def __init__(self,config):
        super(BertForSeq,self).__init__(config)
        self.config=BertConfig(config)
        self.num_labels=config.num_labels #set to 1, it's a logit
        self.bert=BertModel(config)   
        self.dropout=nn.Dropout(config.hidden_derpout_prob)
        self.classifier=nn.Linear(config._hidden_size,self.num_labels)
        
        self.init_weights()
        
    def forward(self,input_ids,attention_mask=None,token_type_ids=None,return_dict=None ):
        return_dict=return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs=self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        ) #prediction
        print(outputs.shape)
        sys.exit()
        pooled_output=outputs[1]
        pooled_output=self.dropout(pooled_output)
        logits=self.classifier(pooled_output)
        return logits