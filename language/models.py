import torch
import numpy as np
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel, GPT2Model, GPT2Tokenizer
from torch import nn

class BertClassifier(torch.nn.Module):

    def __init__(self, dropout=0.5, num_classes=6):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)
        #self.relu = nn.ReLU()

    def forward(self, input_id, mask, ret_rep = 0):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        if(ret_rep == 2):
            inter = pooled_output
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        if(ret_rep == 1):
            inter = linear_output
        #final_layer = self.relu(linear_output)

        if(ret_rep == 0):
            return linear_output
        else:
            return linear_output, inter

class DistilBertClassifier(torch.nn.Module):

    def __init__(self, dropout=0.5, num_classes=6):

        super(DistilBertClassifier, self).__init__()

        self.bert = DistilBertModel.from_pretrained('distilbert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)
        #self.relu = nn.ReLU()

    def forward(self, input_id, mask, ret_rep = 0):

        output1 = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        hidden_state = output1[0]
        pooled_output = hidden_state[:, 0]
        if(ret_rep == 2):
            inter = pooled_output
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        if(ret_rep == 1):
            inter = linear_output
        #final_layer = self.relu(linear_output)

        if(ret_rep == 0):
            return linear_output
        else:
            return linear_output, inter

class GPT2Classifier(torch.nn.Module):

    def __init__(self, dropout=0.5, num_classes=6):

        super(GPT2Classifier, self).__init__()

        self.bert = GPT2Model.from_pretrained('gpt2')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)
        #self.relu = nn.ReLU()

    def forward(self, input_id, mask, ret_rep = 0):

        output1 = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        hidden_state = output1[0]
        pooled_output = hidden_state[:, 0]
        if(ret_rep == 2):
            inter = pooled_output
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        if(ret_rep == 1):
            inter = linear_output
        #final_layer = self.relu(linear_output)

        if(ret_rep == 0):
            return linear_output
        else:
            return linear_output, inter