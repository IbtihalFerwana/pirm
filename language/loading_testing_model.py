import pandas as pd
import torch
import numpy as np
#from torch._C import int8
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import json
import glob
from torch import autograd
import time
import argparse
import random
import torch
import copy
from collections import OrderedDict


class Dataset(torch.utils.data.Dataset):


    def __init__(self, train_data, tokenizer, nlp_task):
        if nlp_task == 'scierc':
            self.labels = train_data[1]
        elif nlp_task == 'aic':
            labels_map = {
            1:0,
            2:1
            }
            self.labels = [labels_map[s] for s in train_data[1]]
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

    def __init__(self, dropout=0.5):

        super(DistilBertClassifier, self).__init__()

        self.bert = DistilBertModel.from_pretrained('distilbert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 6)
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
        
    
def evaluate(model, val_loader, use_cuda):
    
    #data_len = len(val_data[0])
    #val_data = Dataset(val_data, tokenizer)

    #val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=2)
    val_dataloader = val_loader

    if use_cuda:
        model = model.cuda()
    
    model.eval()
    
    
    total_acc_val = 0
    total_loss_val = 0
    sample_count = 0

    criterion = nn.CrossEntropyLoss(reduction='mean')

    with torch.no_grad():

      for val_input, val_label in val_dataloader:

        val_label = val_label.to(device)
        mask = val_input['attention_mask'].to(device)
        input_id = val_input['input_ids'].squeeze(1).to(device)

        output = model(input_id, mask)

        batch_loss = criterion(output, val_label)
        total_loss_val += batch_loss.item()*val_label.size(0)
        
        acc = (output.argmax(dim=1) == val_label).sum().item()
        total_acc_val += acc
        sample_count += val_label.size(0)

        ### save memory
        val_label.to("cpu")
        mask.to("cpu")
        input_id.to("cpu")

    #print(f'Validation Accuracy: {total_acc_val / sample_count: .3f}')    
    return  total_loss_val/sample_count, total_acc_val/sample_count


def mean_accuracy(logits, y):

    acc = (logits.argmax(dim=1) == y).int().float().mean()
    return acc    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='time alignment -- BERT IRM')

    
    parser.add_argument('--data_dir', type=str, default='data', help='data directory')
    parser.add_argument('--data_split', type=str, default='equal_split', help='define env split')
    parser.add_argument('--testing_years', nargs='+', default = ['1980','1990','2000','2005','2010'], help='test list of partitioned periods, indicated by the starting year')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--model_path', type=str, default='saved_models/', help='checkpoints')
    parser.add_argument('--nlp_task', type=str, default='scierc', help='nlp tasks: scierc, aic')
    parser.add_argument('--ib', type=int, default=0, help='with ib_lambda value')

    

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    testing_years = args.testing_years
    data_dir = args.data_dir
    data_split = args.data_split
    batch_size = args.batch_size
    task = args.nlp_task

    
    criterion = nn.CrossEntropyLoss()
    # optimizer = Adam(model.parameters(), lr= learning_rate, eps=1e-08)
    tokenizer_name = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    
    model_path = args.model_path
    # model.eval()
    res = []
    envs_names = ["envs3","envs2"]
    # envs_names = ["envs4"]
    # envs_years = [["1980","1990","2000","2005"],["1990","2000","2005"],["2000","2005"]]
    envs_years = [["1990","2000","2005"],["2000","2005"]]
    # envs_years = [["1980","1990","2000","2005"]]
    # envs_years = [["2006","2009","2012", "2015"], ["2009", "2012", "2015"],["2012","2015"]]
    # envs_years = [["2009", "2012", "2015"],["2012","2015"]]
    # envs_years = [["2006","2009","2012", "2015"]]
    # seeds = ["200","300"]
    seeds = ["100","200","300"]
    # seeds = ["100"]
    # pen = '1000'
    # anneal = '10'
    # anneals = [str(i) for i in [10,15,20]]
    # anneals = [str(i) for i in [20,30,35,40]]
    # pens = [str(i) for i in [100,1000,10000]]
    anneals = [str(i) for i in [40]]
    # anneals = [str(i) for i in [20,30,35,40]]
    pens = [str(i) for i in [100]]
    # seeds = ["100"]
    # envs = ["envs4"]
    # ibs = ["0.01","0.1", "1", "10", "50"]
    ibs = ["1"]

    if args.ib == 0:
        ibs = ["-1"]

    if args.ib > 0:
        for i, env_name in enumerate(envs_names):
            training_years = envs_years[i]
            print("\ttraining years: ", training_years)
            for pen in pens:
                print("\tpenalty: ", pen)
                for anneal in anneals:
                    print("\tanneal: ", anneal)
                    for seed in seeds:
                        print("\tseed ", seed)
                        random.seed(seed)
                        torch.manual_seed(seed)
                        for ib in ibs:
                            if args.ib > 0:
                                PATH = f'{model_path}/{env_name}_pen_{pen}_anneal_{anneal}_seed_{seed}_ib_{ib}_best_model_ckpt'
                            else:
                                PATH = f'{model_path}/{env_name}_pen_{pen}_anneal_{anneal}_seed_{seed}_best_model_ckpt'

                            print("\t",PATH)
                            checkpoint = torch.load(PATH)
                            epoch = checkpoint['epoch']

                            state_dict = checkpoint['model_state_dict']
                            
                            new_state_dict = OrderedDict()

                            for k, v in state_dict.items():
                                # if 'module' not in k:
                                if 'module' in k:
                                    k = '.'.join(k.split('.')[1:])
                                # else:
                                    # k = k.replace('features.module.', 'module.features.')
                                new_state_dict[k]=v                    

                            print("\tEPOCH: ", epoch)
                            if task == 'scierc':
                                num_classes = 6
                            elif task == 'aic':
                                num_classes = 2
                            model = BertClassifier(num_classes=num_classes)
                            model.load_state_dict(new_state_dict)
                            # model.load_state_dict(checkpoint['model_state_dict'])
                            
                            training_data = []
                            training_label = []
                            val_files = []
                            env_sizes = []

                            for i, yr in enumerate(training_years): 
                                print(yr)
                                
                                g = glob.glob("{}/{}/train/{}*".format(data_dir, data_split, yr))
                                train_file_i = g[0]
                                text_list = []
                                labels_list = []
                                with open(train_file_i, 'r') as f:
                                    lines = f.readlines()
                                    for line in lines:
                                        d = json.loads(line)

                                        text_list.append(d["text"])
                                        labels_list.append(d["labels"])

                                print("env size: ", len(text_list))
                                env_sizes.append(len(text_list))

                                training_data.append(text_list)
                                training_label.append(labels_list)
                                    

                                g_val = glob.glob("{}/{}/val/{}*".format(data_dir, data_split, yr))
                                val_files.append(g_val[0])
                            
                            #### build train environments

                            envs = [{} for i in range(len(training_data))]
                            i = 0

                            for text_list, labels_list in zip(training_data, training_label):
                                train_data = [text_list, labels_list]
                                train = Dataset(train_data, tokenizer,task)
                                train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
                                envs[i]["train_dataloader"] = train_dataloader
                                
                                envs[i]["train"] = True
                                envs[i]["year"] = training_years[i]

                                i+=1

                            training_accs=[]

                            for env in envs:
                                
                                mean_train_loss, mean_train_acc = evaluate(model, env["train_dataloader"], use_cuda)

                                training_accs.append(mean_train_acc)
                                d = {"train_period":env["year"],"train_accuracy":mean_train_acc}
                                print(d)
                            
                            # all periods tested together (overall accuracy)
                            if len(envs) > 1:
                                print("all training periods")

                                d = {"train_period":','.join([env["year"] for env in envs]), "mean_train_acc":np.mean(training_accs)}
                                print(d)
                            
                            steps = np.max(env_sizes)//batch_size

                            i = 0
                            for val_file in val_files:
                                text_list_val = []
                                labels_list_val = []
                                with open(val_file, 'r') as f:
                                    lines = f.readlines()
                                    for line in lines:
                                        d = json.loads(line)
                                        text_list_val.append(d["text"])
                                        labels_list_val.append(d["labels"])
                                val_data = [text_list_val, labels_list_val]
                                val_data = Dataset(val_data, tokenizer,task)
                            
                                val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
                                envs[i]["val_dataloader"] = val_dataloader
                                i+=1

                            val_dataloaders = [envs[i]["val_dataloader"] for i in range(len(envs)) if envs[i]['train']==True]
                            epoch_val_loss = []
                            epoch_val_acc = []
                            for val_dataloader in val_dataloaders:
                                val_loss, val_acc = evaluate(model, val_dataloader, use_cuda)
                                epoch_val_loss.append(val_loss)
                                epoch_val_acc.append(val_acc)

                            ### avg
                            avg_val_acc = np.mean(epoch_val_acc)
                            avg_val_loss = np.mean(epoch_val_loss)
                            ### worst
                            min_val_acc = np.min(epoch_val_acc)

                            print("\tAvg. validation: ", avg_val_acc)
                            testing_accs = []
                            test_years = []

                            for yr_test in testing_years:
                                g = glob.glob("{}/{}/test/{}*".format(data_dir, data_split, yr_test))

                                print(yr_test)

                                test_file_i = g[0]

                                test_text_list = []
                                test_labels_list = []

                                with open(test_file_i, 'r') as f:
                                    lines = f.readlines()
                                    for line in lines:
                                        d = json.loads(line)
                                        test_text_list.append(d["text"])
                                        test_labels_list.append(d["labels"])

                                
                                print("\tLEN OF TEST DATA: ", len(test_text_list))
                                test_data = [test_text_list, test_labels_list]

                                test_data = Dataset(test_data, tokenizer,task)

                                test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
                                
                                test_loss, test_acc = evaluate(model, test_dataloader, use_cuda)
                                # test_acc = 0

                                testing_accs.append(test_acc) 

                                d = {"test_period":yr_test,"test_acc":test_acc}
                                print("\t",d)
                                test_years.append(yr_test)
                
                            # all periods tested together (overall accuracy)
                            if len(testing_years) > 1:
                                #test_data = [all_test_text, all_test_label]
                                print("all testing periods")

                                d = {"test_period":','.join(testing_years), "test_acc":np.mean(testing_accs)}
                                print(d)
                            
                            if args.ib > 0:
                                df = pd.DataFrame(list(zip(test_years, testing_accs)),\
                                                            columns =['test_year', 'test_avg_acc'])
                                df.to_pickle(f"{args.model_path}/{env_name}_pen_{pen}_anneal_{anneal}_seed_{seed}_ib_{ib}_best_model_test_results")

                                df = pd.DataFrame(list(zip([epoch],[np.mean(training_accs)],[avg_val_acc],[min_val_acc],[test_years], [testing_accs])),\
                                                            columns =['epoch','avg_train_acc','avg_val_acc','min_val','test_year', 'test_avg_acc'])
                                df.to_pickle(f"{args.model_path}/{env_name}_pen_{pen}_anneal_{anneal}_seed_{seed}_ib_{ib}_best_model_all_results")
                            else:
                                df = pd.DataFrame(list(zip(test_years, testing_accs)),\
                                                    columns =['test_year', 'test_avg_acc'])
                                df.to_pickle(f"{args.model_path}/{env_name}_pen_{pen}_anneal_{anneal}_seed_{seed}_best_model_test_results")

                                df = pd.DataFrame(list(zip([epoch],[np.mean(training_accs)],[avg_val_acc],[min_val_acc],[test_years], [testing_accs])),\
                                                            columns =['epoch','avg_train_acc','avg_val_acc','min_val','test_year', 'test_avg_acc'])
                                df.to_pickle(f"{args.model_path}/{env_name}_pen_{pen}_anneal_{anneal}_seed_{seed}_best_model_all_results")

        
    # else:


    #     for i, env_name in enumerate(envs_names):
    #         training_years = envs_years[i]
    #         print("\ttraining years: ", training_years)
    #         for pen in pens:
    #             print("\tpenalty: ", pen)
    #             for anneal in anneals:
    #                 print("\tanneal: ", anneal)
    #                 for seed in seeds:
    #                     print("\tseed ", seed)
    #                     random.seed(seed)
    #                     torch.manual_seed(seed)
                        
                        
    #                     print("\t",PATH)
    #                     checkpoint = torch.load(PATH)
    #                     epoch = checkpoint['epoch']

    #                     state_dict = checkpoint['model_state_dict']
                        
    #                     new_state_dict = OrderedDict()

    #                     for k, v in state_dict.items():
    #                         # if 'module' not in k:
    #                         if 'module' in k:
    #                             k = '.'.join(k.split('.')[1:])
    #                         # else:
    #                             # k = k.replace('features.module.', 'module.features.')
    #                         new_state_dict[k]=v                    

    #                     print("\tEPOCH: ", epoch)
    #                     if task == 'scierc':
    #                         num_classes = 6
    #                     elif task == 'aic':
    #                         num_classes = 2
    #                     model = BertClassifier(num_classes=num_classes)
    #                     model.load_state_dict(new_state_dict)
    #                     # model.load_state_dict(checkpoint['model_state_dict'])
                        
    #                     training_data = []
    #                     training_label = []
    #                     val_files = []
    #                     env_sizes = []

    #                     for i, yr in enumerate(training_years): 
    #                         print(yr)
                            
    #                         g = glob.glob("{}/{}/train/{}*".format(data_dir, data_split, yr))
    #                         train_file_i = g[0]
    #                         text_list = []
    #                         labels_list = []
    #                         with open(train_file_i, 'r') as f:
    #                             lines = f.readlines()
    #                             for line in lines:
    #                                 d = json.loads(line)

    #                                 text_list.append(d["text"])
    #                                 labels_list.append(d["labels"])

    #                         print("env size: ", len(text_list))
    #                         env_sizes.append(len(text_list))

    #                         training_data.append(text_list)
    #                         training_label.append(labels_list)
                                

    #                         g_val = glob.glob("{}/{}/val/{}*".format(data_dir, data_split, yr))
    #                         val_files.append(g_val[0])
                        
    #                     #### build train environments

    #                     envs = [{} for i in range(len(training_data))]
    #                     i = 0

    #                     for text_list, labels_list in zip(training_data, training_label):
    #                         train_data = [text_list, labels_list]
    #                         train = Dataset(train_data, tokenizer,task)
    #                         train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    #                         envs[i]["train_dataloader"] = train_dataloader
                            
    #                         envs[i]["train"] = True
    #                         envs[i]["year"] = training_years[i]

    #                         i+=1

    #                     training_accs=[]

    #                     for env in envs:
                            
    #                         mean_train_loss, mean_train_acc = evaluate(model, env["train_dataloader"], use_cuda)

    #                         training_accs.append(mean_train_acc)
    #                         d = {"train_period":env["year"],"train_accuracy":mean_train_acc}
    #                         print(d)
                        
    #                     # all periods tested together (overall accuracy)
    #                     if len(envs) > 1:
    #                         print("all training periods")

    #                         d = {"train_period":','.join([env["year"] for env in envs]), "mean_train_acc":np.mean(training_accs)}
    #                         print(d)
                        
    #                     steps = np.max(env_sizes)//batch_size

    #                     i = 0
    #                     for val_file in val_files:
    #                         text_list_val = []
    #                         labels_list_val = []
    #                         with open(val_file, 'r') as f:
    #                             lines = f.readlines()
    #                             for line in lines:
    #                                 d = json.loads(line)
    #                                 text_list_val.append(d["text"])
    #                                 labels_list_val.append(d["labels"])
    #                         val_data = [text_list_val, labels_list_val]
    #                         val_data = Dataset(val_data, tokenizer,task)
                        
    #                         val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    #                         envs[i]["val_dataloader"] = val_dataloader
    #                         i+=1

    #                     val_dataloaders = [envs[i]["val_dataloader"] for i in range(len(envs)) if envs[i]['train']==True]
    #                     epoch_val_loss = []
    #                     epoch_val_acc = []
    #                     for val_dataloader in val_dataloaders:
    #                         val_loss, val_acc = evaluate(model, val_dataloader, use_cuda)
    #                         epoch_val_loss.append(val_loss)
    #                         epoch_val_acc.append(val_acc)

    #                     ### avg
    #                     avg_val_acc = np.mean(epoch_val_acc)
    #                     avg_val_loss = np.mean(epoch_val_loss)
    #                     ### worst
    #                     min_val_acc = np.min(epoch_val_acc)

    #                     print("\tAvg. validation: ", avg_val_acc)
    #                     testing_accs = []
    #                     test_years = []

    #                     for yr_test in testing_years:
    #                         g = glob.glob("{}/{}/test/{}*".format(data_dir, data_split, yr_test))

    #                         print(yr_test)

    #                         test_file_i = g[0]

    #                         test_text_list = []
    #                         test_labels_list = []

    #                         with open(test_file_i, 'r') as f:
    #                             lines = f.readlines()
    #                             for line in lines:
    #                                 d = json.loads(line)
    #                                 test_text_list.append(d["text"])
    #                                 test_labels_list.append(d["labels"])

                            
    #                         print("\tLEN OF TEST DATA: ", len(test_text_list))
    #                         test_data = [test_text_list, test_labels_list]

    #                         test_data = Dataset(test_data, tokenizer,task)

    #                         test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
                            
    #                         test_loss, test_acc = evaluate(model, test_dataloader, use_cuda)
    #                         # test_acc = 0

    #                         testing_accs.append(test_acc) 

    #                         d = {"test_period":yr_test,"test_acc":test_acc}
    #                         print("\t",d)
    #                         test_years.append(yr_test)
            
    #                     # all periods tested together (overall accuracy)
    #                     if len(testing_years) > 1:
    #                         #test_data = [all_test_text, all_test_label]
    #                         print("all testing periods")

    #                         d = {"test_period":','.join(testing_years), "test_acc":np.mean(testing_accs)}
    #                         print(d)
                        

    #                     df = pd.DataFrame(list(zip(test_years, testing_accs)),\
    #                                                 columns =['test_year', 'test_avg_acc'])
    #                     df.to_pickle(f"{args.model_path}/{env_name}_pen_{pen}_anneal_{anneal}_seed_{seed}_best_model_test_results")

    #                     df = pd.DataFrame(list(zip([epoch],[np.mean(training_accs)],[avg_val_acc],[min_val_acc],[test_years], [testing_accs])),\
    #                                                 columns =['epoch','avg_train_acc','avg_val_acc','min_val','test_year', 'test_avg_acc'])
    #                     df.to_pickle(f"{args.model_path}/{env_name}_pen_{pen}_anneal_{anneal}_seed_{seed}_best_model_all_results")
