import pandas as pd
import torch
import numpy as np
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
        

# def evaluate(model, test_data, tokenizer, use_cuda, device):

#     test = Dataset(test_data, tokenizer)

#     test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

#     if use_cuda:

#         model = model.cuda()
#         model.eval()

#     total_acc_test = 0
#     with torch.no_grad():

#         for test_input, test_label in test_dataloader:

#             test_label = test_label.to(device)
#             mask = test_input['attention_mask'].to(device)
#             input_id = test_input['input_ids'].squeeze(1).to(device)

#             output = model(input_id, mask)

#             acc = (output.argmax(dim=1) == test_label).sum().item()
#             total_acc_test += acc
    
#     print(f'Test Accuracy: {total_acc_test / len(test_data[1]): .3f}')
#     return total_acc_test / len(test_data[1])

    
def evaluate(model, val_loader, tokenizer, use_cuda):
    
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

    #print(f'Validation Accuracy: {total_acc_val / sample_count: .3f}')    
    return  total_loss_val/sample_count, total_acc_val/sample_count


def penalty(logits, y, criterion):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = criterion(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)


def mean_accuracy(logits, y):

    acc = (logits.argmax(dim=1) == y).int().float().mean()
    return acc    



def train_model(n_steps, envs, model, val_dataloaders, tokenizer, optim, args, method='erm'):

    l2_regularizer_weight = args.l2_regularizer
    p_weight = args.penalty_weight
    penalty_anneal_iters = args.penalty_anneal_iters

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    
    stime = time.time()

    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    optimizer = optim
    
    steps = n_steps
    epochs = args.epochs

    
    dataloaders = [envs[i]["dataloader"] for i in range(len(envs))]
    
    d_num = len(dataloaders)
    
    val_acc_ls = []
    val_loss_ls = []
    train_acc_ls = []
    train_loss_ls = [] 

    print("number of envs: ", d_num)
    max_val = 0
    model_path = args.model_path
    
    model.train()

    for epoch in range(epochs):
        iters = [iter(d) for d in dataloaders]
        
        val_acc_ls_epoch = []
        train_acc_ls_epoch = []
        train_loss_ls_epoch = []

        #acc = 0
        #losses = [0]*d_num
        #pens = [0]*d_num

        if(epoch == 0):
            print("Training Begins")    
            print("Method: ", args.method)
            print("Information Bottleneck Penalty: ", args.ib_lambda > 0)
            print("batch size: ", args.batch_size)

        print("Epoch: ", epoch, ", Number of Iterations: ", steps)    
        
        for step in range(steps):
            # print("Step: ", step)

            for i in range(d_num):

                #t1 = iters[i].next()
                ### resample if needed
                try:
                    current_batch = next(iters[i])
                except StopIteration:

                    iters[i] = iter(dataloaders[i])
                    current_batch = next(iters[i])

                train_label_0 = current_batch[1]
                train_input_0 = current_batch[0]
                train_label = train_label_0.to(device)
                mask = train_input_0['attention_mask'].to(device)
                input_id = train_input_0['input_ids'].squeeze(1).to(device)

                if args.inter != 0:
                    logits, inter_logits = model(input_id, mask, args.inter)
                else:
                    logits = model(input_id, mask)

                #env['nll'] = mean_nll(logits, env['labels'][val_size:])
                envs[i]['loss'] = criterion(logits, train_label)
                envs[i]['acc'] = mean_accuracy(logits, train_label)
                envs[i]['penalty'] = penalty(logits, train_label, criterion)
                
                if args.ib_lambda > 0.:
                    if args.class_condition:
                        num_classes = args.num_classes
                        index = [train_label.squeeze() == j for j in range(num_classes)]
                        envs[i]['var'] = sum(inter_logits[ind].var(dim=0).mean() for ind in index)
                        envs[i]['var'] /= num_classes
                    else:
                        envs[i]['var'] = inter_logits.var(dim=0).mean()


            train_loss = torch.stack([envs[i]['loss'] for i in range(d_num) if envs[i]['train']==True]).mean()
            train_loss_ls_epoch.append(train_loss.item())

            train_acc = torch.stack([envs[i]['acc'] for i in range(d_num) if envs[i]['train']==True]).mean()

            train_acc_ls_epoch.append(train_acc.item())
            # train_acc_ls.append(train_acc.item())

        
            weight_norm = torch.tensor(0.).cuda()
            for w in model.parameters():
                weight_norm += w.norm().pow(2)

            train_loss+=(l2_regularizer_weight*weight_norm)

            ### error
            #train_loss+=(l2_regularizer_weight)

            if(args.ib_lambda > 0.):

                ib_weight = args.ib_lambda if step >= args.ib_step else 0.
                var_loss = torch.stack([envs[i]['loss'] for i in range(d_num) if envs[i]['train']==True]).mean()
                train_loss += ib_weight * var_loss

            if method == 'irm':
                
                train_penalty = torch.stack([envs[i]['loss'] for i in range(d_num) if envs[i]['train']==True]).mean()
                penalty_weight = (p_weight if step >= penalty_anneal_iters else .0)
                train_loss += penalty_weight * train_penalty
                if penalty_weight > 1.0:
                    train_loss /= penalty_weight
            elif method == 'erm':
                penalty_weight = 0.

                ### set penalty to 0
                train_penalty = envs[0]['penalty'] * 0.  # so that this term is a tensor
            else:
                raise NotImplementedError    


            model.zero_grad()
            train_loss.backward()
            optimizer.step()

            if step % (steps//5) == 0:
                print(f'Steps: {step+1} | Training Loss: {sum(train_loss_ls_epoch)/len(train_loss_ls_epoch):.3f} | \
                            Training Accuracy: {sum(train_acc_ls_epoch)/len(train_acc_ls_epoch):.3f}')

        
        
        epoch_train_loss = sum(train_loss_ls_epoch)/len(train_loss_ls_epoch)
        epoch_train_acc = sum(train_acc_ls_epoch)/len(train_acc_ls_epoch)
        
        train_loss_ls.append(epoch_train_loss)
        train_acc_ls.append(epoch_train_acc)

        print("###### traing acc: ",epoch_train_acc)
        

        ### validate model
        epoch_val_acc = []
        epoch_val_loss = []
        
        for val_dataloader in val_dataloaders:
            val_loss, val_acc = evaluate(model, val_dataloader, tokenizer, use_cuda)
            epoch_val_loss.append(val_loss)
            epoch_val_acc.append(val_acc)
        avg_val = np.mean(epoch_val_acc)
        avg_loss = np.mean(epoch_val_loss)
        val_acc_ls.append(avg_val)
        val_loss_ls.append(np.mean(epoch_val_loss))
        output_cols = []
        if (avg_val - 0.01) > max_val:
            max_val = avg_val
            model_path_pt = model_path+ '_checkpoint.pt'
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            }, model_path_pt)
            if(len(epoch_val_acc) > 0):
                envs_cols = [epoch_train_loss,epoch_train_acc]
                envs_cols_names = ["train_loss", "train_acc"]
                
                for jj in range(len(epoch_val_acc)):
                    envs_cols_names.append("val_loss_env_{}".format(jj))
                    envs_cols_names.append("val_acc_env_{}".format(jj))
                    envs_cols.append(epoch_val_loss[jj])
                    envs_cols.append(epoch_val_acc[jj])
                    
                best_results = pd.DataFrame(envs_cols, index=envs_cols_names).T
            else:
                envs_cols = [epoch_train_loss,epoch_train_acc]
                envs_cols_names = ["train_loss", "train_acc"]
                best_results = pd.DataFrame(envs_cols, index=envs_cols_names).T
        if epoch == 0:
            if len(epoch_val_acc) == 2:
                all_env_val_acc = [[],[]]
                all_env_val_loss = [[],[]]
            elif len(epoch_val_acc) == 4:
                all_env_val_acc = [[],[],[],[]]
                all_env_val_loss = [[],[],[],[]]
            if(len(epoch_val_acc) > 0):                
                for jj in range(len(epoch_val_acc)):
                    all_env_val_loss[jj].append(epoch_val_loss[jj])
                    all_env_val_acc[jj].append(epoch_val_acc[jj])
        else:
            if(len(epoch_val_acc) > 0):                
                for jj in range(len(epoch_val_acc)):
                    print("env: ", epoch_val_loss)
                    print("env epoch val: ", epoch_val_loss[jj])
                    all_env_val_loss[jj].append(epoch_val_loss[jj])
                    all_env_val_acc[jj].append(epoch_val_acc[jj])


        model.train()

        
        print(f'Epoch: {epoch} | Training Loss: {sum(train_loss_ls_epoch)/len(train_loss_ls_epoch):.3f} | \
                    Training Accuracy: {sum(train_acc_ls_epoch)/len(train_acc_ls_epoch):.3f}')
        for i, val_acc_i in enumerate(epoch_val_acc):
            print(f'Validation Accuracy for env # {i}: {val_acc_i:.3f} ')
        
    

    return train_loss_ls, train_acc_ls, all_env_val_loss, all_env_val_acc, best_results
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='time alignment -- BERT IRM')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory')
    parser.add_argument('--training_years', nargs='+', default = ['1980','1990','2000','2005','2010','2015'], help='a list of partitioned periods, indicated by the starting year')
    parser.add_argument('--testing_years', nargs='+', default = ['1980','1990','2000','2005','2010','2015'], help='a list of partitioned periods, indicated by the starting year')
    parser.add_argument('--output_file',  default = 'data/output_erm', help='output file to save results')
    parser.add_argument('--method',  type = str, default = "erm" , help='ERM or IRM')
    parser.add_argument('--epochs',  type = int, default = 50, help='number of training epochs')
    parser.add_argument('--learning_rate',  type = float, default =1e-6 , help='learning rate')
    parser.add_argument('--l2_regularizer',  type = float, default =0.001 , help='l2_regularizer_weight for IRM')
    parser.add_argument('--penalty_weight',  type = float, default = 0.1 , help='penalty_weight for IRM')
    parser.add_argument('--penalty_anneal_iters',  type = float, default =10 , help='penalty_anneal_iters for IRM')
    parser.add_argument('--inter',  type = int, default = 2 , help='specify layer for imposing bottleneck penalty')
    parser.add_argument('--ib_lambda',  type = float, default = 0.1 , help='IB penalty weight')
    parser.add_argument('--class_condition',  type = float, default = False , help='IB penalty classwise application')
    parser.add_argument('--ib_step',  type = int, default = 10 , help='penalty_anneal_iters for IB')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--model_path', type=str, default='saved_models/', help='checkpoints')
    parser.add_argument('--seed', type=int, default=0, help='seed number')
    parser.add_argument('--model', type=str, default='bert', help='language model for finetuning, bert or distilbert')

    
    args = parser.parse_args()
    
    training_years = args.training_years
    testing_years = args.testing_years
    data_dir = args.data_dir
    output_file = args.output_file
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    model_name = args.model

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    
    stime = time.time()
    files = []
    val_files = []
    test_res = []
    dataloaders = []
    
    criterion = nn.CrossEntropyLoss()


    random.seed(args.seed)
    #### read data
    
    training_data = []
    training_label = []
    lengths = []
    val_files = []

    train_period = '_'.join(training_years)
    if model_name == 'bert':
        model = BertClassifier()
        tokenizer_name = 'bert-base-cased'
    elif model_name == 'distilbert':
        model = DistilBertClassifier()
        tokenizer_name = 'distilbert-base-cased'
    else:
        raise NotImplementedError
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    model = model.cuda()
    optimizer = Adam(model.parameters(), lr= learning_rate, eps=1e-08)
    env_sizes = []

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

        print("env size: ", len(text_list))
        env_sizes.append(len(text_list))

        training_data.append(text_list)
        training_label.append(labels_list)
            

        g_val = glob.glob("{}/dev/{}*".format(data_dir, yr))
        val_files.append(g_val[0])
    
    #### build train environments

    envs = [{} for i in range(len(training_data))]

    i = 0

    for text_list, labels_list in zip(training_data, training_label):
        train_data = [text_list, labels_list]
        train = Dataset(train_data, tokenizer)
        train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        envs[i]["dataloader"] = train_dataloader

        envs[i]["train"] = True
        envs[i]["year"] = training_years[i]
        i+=1

    #### add validation environments here
    
    # envs = envs + valid_envs    

    steps = np.max(env_sizes)//batch_size

    
    val_dataloaders = []
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
        val_data = Dataset(val_data, tokenizer)
    
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
        val_dataloaders.append(val_dataloader)
    #val_loss, val_acc = validate(model, val_dataloader, data_len, tokenizer, use_cuda, device)


    train_loss_ls, train_acc_ls, val_loss_ls, val_acc_ls, best_results = train_model(steps, envs, model, val_dataloaders, tokenizer, optimizer, args, args.method)

    
    if(len(val_acc_ls) > 0):
        cols_name = ["Train Loss", "Train Accuracy"]
        cols = [train_loss_ls,train_acc_ls]
        for cc, (env_val_loss, env_val_acc) in enumerate(zip(val_loss_ls, val_acc_ls)):
            cols.append(env_val_loss)
            cols.append(env_val_acc)
            cols_name.append("Validation Loss Env "+str(cc))
            cols_name.append("Validation Accuracy Env "+str(cc))
        # train_history = pd.DataFrame(list(zip(cols)), \
                                                    # columns = cols_name)
        train_history = pd.DataFrame(cols).T
        print(train_history)
        train_history.columns = cols_name
        


        # train_history = pd.DataFrame(list(zip(train_loss_ls, train_acc_ls, val_loss_ls, val_acc_ls)), \
        #                                             columns = ["Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy"])
    else:
        train_history = pd.DataFrame(list(zip(train_loss_ls, train_acc_ls)), columns = ["Loss", "Accuracy"])    


    ### get final training accuracy

    overall_count = 0
    overall_acc = 0

    for env in envs:
        
        train_loss, train_acc = evaluate(model, env["dataloader"], tokenizer, use_cuda)
        env_len = len(env["dataloader"].dataset)

        overall_count += env_len
        overall_acc += env_len*train_acc 

        d = {"train_period":env["year"],"train_accuracy":train_acc}
        print(d)
        #all_res.append(d)
    
    # all periods tested together (overall accuracy)
    if len(envs) > 1:
        print("all training periods")

        d = {"train_period":','.join([env["year"] for env in envs]), "train_acc":overall_acc/overall_count}
        print(d)
        #all_res.append(d)



    ### get final testing accuracy

    overall_count = 0
    overall_acc = 0

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
        
        #all_test_text.extend(test_text_list)
        #all_test_label.extend(test_labels_list)

        test_data = Dataset(test_data, tokenizer)

        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
        
        test_loss, test_acc = evaluate(model, test_dataloader, tokenizer, use_cuda)
        env_len = len(test_labels_list)

        overall_count += env_len
        overall_acc += env_len*test_acc 

        d = {"test_period":yr_test,"test_acc":test_acc}
        print(d)
        test_res.append(d)
    
    # all periods tested together (overall accuracy)
    if len(testing_years) > 1:
        #test_data = [all_test_text, all_test_label]
        print("all testing periods")
        #test_acc = evaluate(model, test_data,tokenizer, use_cuda, device)

        d = {"test_period":','.join(testing_years), "test_acc":overall_acc/overall_count}
        print(d)
        test_res.append(d)
    
    #pd.DataFrame(all_res).to_csv(output_file+"_combinded.csv")
    pd.DataFrame(train_history).to_csv(output_file+"_train_history.csv")
    best_results.to_csv(output_file+"_best_train_val_results.csv")
    pd.DataFrame(test_res).to_csv(output_file+"_test_results.csv")
    
    etime=time.time()
    print("time: ", (etime-stime)/60)