import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import json
import glob
from torch import autograd
import time
import argparse


class Dataset(torch.utils.data.Dataset):

    def __init__(self, train_data, tokenizer):

        self.labels = train_data[1]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in train_data[0]]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class BertClassifier(torch.nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 6)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer
        
def evaluate(model, test_data, tokenizer, use_cuda, device):

    test = Dataset(test_data, tokenizer)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    if use_cuda:

        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    
    print(f'Test Accuracy: {total_acc_test / len(test_data[1]): .3f}')
    return total_acc_test / len(test_data[1])
    
def validate(model, val_data, tokenizer, use_cuda, device):
    
    data_len = len(val_data[0])
    val_data = Dataset(val_data, tokenizer)

    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=2)

    if use_cuda:
        model = model.cuda()
    
    
    total_acc_val = 0
    total_loss_val = 0

    with torch.no_grad():

      for val_input, val_label in val_dataloader:

        val_label = val_label.to(device)
        mask = val_input['attention_mask'].to(device)
        input_id = val_input['input_ids'].squeeze(1).to(device)

        output = model(input_id, mask)

        batch_loss = criterion(output, val_label)
        total_loss_val += batch_loss.item()
        
        acc = (output.argmax(dim=1) == val_label).sum().item()
        total_acc_val += acc
    return  total_loss_val/data_len, total_acc_val/data_len

def penalty(logits, y, criterion):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = criterion(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='time alignment -- BERT IRM')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory')
    parser.add_argument('--training_years', nargs='+', default = ['1980','1990','2000','2005','2010','2015'], help='a list of partitioned periods, indicated by the starting year')
    parser.add_argument('--testing_years', nargs='+', default = ['1980','1990','2000','2005','2010','2015'], help='a list of partitioned periods, indicated by the starting year')
    parser.add_argument('--output_file',  default = 'data/output_erm.csv', help='output file to save results')
    parser.add_argument('--epochs',  type = int, default = 50, help='number of training epochs')
    parser.add_argument('--learning_rate',  type = float, default =1e-6 , help='learning rate')
    parser.add_argument('--l2_regularizer',  type = float, default =0.001 , help='l2_regularizer_weight for IRM')
    parser.add_argument('--penalty_weight',  type = float, default =100 , help='penalty_weight for IRM')
    parser.add_argument('--penalty_anneal_iters',  type = float, default =10 , help='penalty_anneal_iters for IRM')

    
    args = parser.parse_args()
    
    training_years = args.training_years
    testing_years = args.testing_years
    data_dir = args.data_dir
    output_file = args.output_file
    epochs = args.epochs
    learning_rate = args.learning_rate
    
    l2_regularizer_weight = args.l2_regularizer
    penalty_weight = args.penalty_weight
    penalty_anneal_iters = args.penalty_anneal_iters
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    
    stime = time.time()
    files = []
    val_files = []
    all_res = []
    dataloaders = []
    
    criterion = nn.CrossEntropyLoss().cuda()

    # num_batches = [2,3]
    
    training_data = []
    training_label = []
    lengths = []
    val_files = []

    train_period = '_'.join(training_years)
    model = BertClassifier()
    model = model.cuda()
    optimizer = Adam(model.parameters(), lr= learning_rate, eps=1e-08)
    
    for i, yr in enumerate(training_years): 
        print(yr)
        
        g = glob.glob("{}/train/{}*".format(data_dir, yr))
        train_file_i = g[0]
        text_list = []
        labels_list = []
        with open(train_file_i, 'r') as f:
            lines = f.readlines()
            for line in lines:
                d = json.loads(line)

                text_list.append(d["text"])
                labels_list.append(d["labels"])
        training_data.append(text_list)
        training_label.append(labels_list)
        lengths.append(len(text_list))

        print(len(text_list))
        
        g_val = glob.glob("{}/dev/{}*".format(data_dir, yr))
        val_files.append(g_val[0])
        
    gcd_num = int(np.gcd.reduce(lengths))
    batch_sizes = np.array(lengths)/gcd_num
    batch_sizes = [int(i) for i in batch_sizes]
    dataloaders = []

    for text_list, labels_list, bs in zip(training_data, training_label, batch_sizes):

        train_data = [text_list, labels_list]
        train = Dataset(train_data, tokenizer)
        train_dataloader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)
        dataloaders.append(train_dataloader)
    steps = gcd_num
    d_num = len(dataloaders)
    print("number of dataloaders: ", d_num)
    
    for epoch in range(epochs):
        iters = [iter(d) for d in dataloaders]

        acc = 0
        losses = [0]*d_num
        pens = [0]*d_num
        
        for step in range(steps):
            # print("Step: ", step)

            for i in range(d_num):
                
                t1 = iters[i].next()
                train_label_0 = t1[1]
                train_input_0 = t1[0]

                train_label = train_label_0.to(device)
                mask = train_input_0['attention_mask'].to(device)
                input_id = train_input_0['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
    #                 outputs[i].append(output)

                env_loss = criterion(output, train_label)
                losses[i] = env_loss
                acc+= (output.argmax(dim=1) == train_label).sum().item()
                pens[i] = penalty(output, train_label, criterion)


            train_loss = torch.stack(losses).mean()
            train_penalty = torch.stack(pens).mean()
        
            weight_norm = torch.tensor(0.).cuda()
            for w in model.parameters():
                weight_norm += w.norm().pow(2)

            train_loss+=(l2_regularizer_weight*weight_norm)
            train_loss+=(l2_regularizer_weight)

            penalty_weight = (penalty_weight if epoch >= penalty_anneal_iters else 1.0)
            train_loss += penalty_weight * train_penalty
            if penalty_weight > 1.0:
              # Rescale the entire loss to keep gradients in a reasonable range
              train_loss /= penalty_weight

            model.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        # Validation step 
        text_list = []
        labels_list = []
        
        for val_file in val_files:
            with open(val_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    d = json.loads(line)
                    text_list.append(d["text"])
                    labels_list.append(d["labels"])
        val_data = [text_list, labels_list]
        val_loss, val_acc = validate(model, val_data,tokenizer, use_cuda, device)

        
        print(f'Epoch: {epoch} | Training Loss: {train_loss:.3f} | Training Accuracy: {acc/(sum(lengths)):.3f} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.3f}')
        
    all_test_text = []
    all_test_label = []
    # testing_all_years = ",".join(testing_years)
    for yr_test in testing_years:
        g = glob.glob("{}/test/{}*".format(data_dir,yr_test))

        print(yr_test)

        test_file_i = g[0]
#             test_files.append(test_file_i)

        test_text_list = []
        test_labels_list = []

#             for test_file_i in test_files:
        with open(test_file_i, 'r') as f:
            lines = f.readlines()
            for line in lines:
                d = json.loads(line)
                test_text_list.append(d["text"])
                test_labels_list.append(d["labels"])
        test_data = [test_text_list, test_labels_list]
        
        all_test_text.extend(test_text_list)
        all_test_label.extend(test_labels_list)
        
        test_acc = evaluate(model, test_data,tokenizer, use_cuda, device)

        d = {"train_period":train_period,"train_acc": acc/(sum(lengths)), "val_acc": val_acc, "test_period":yr_test,"test_acc":test_acc}
        all_res.append(d)
    
    # all periods tested together (overall accuracy)
    if len(testing_years) > 1:
        test_data = [all_test_text, all_test_label]
        print("all testing periods")
        test_acc = evaluate(model, test_data,tokenizer, use_cuda, device)

        d = {"train_period":train_period, "train_acc": acc/(sum(lengths)), "val_acc": val_acc, "test_period":','.join(testing_years), "test_acc":test_acc}
        all_res.append(d)
    
    pd.DataFrame(all_res).to_csv(output_file)
    
    etime=time.time()
    print("time: ", (etime-stime)/60)