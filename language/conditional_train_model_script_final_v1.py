import pandas as pd
import torch
import numpy as np
#from torch._C import int8
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel, GPT2Model, GPT2Tokenizer
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


def penalty(logits, y, criterion):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = criterion(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)


def mean_accuracy(logits, y):

    acc = (logits.argmax(dim=1) == y).int().float().mean()
    return acc    



def train_model(n_steps, envs, model, optim, args, method='erm', linear_probing = False):

    l2_regularizer_weight = args.l2_regularizer
    p_weight = args.penalty_weight
    penalty_anneal_iters = args.penalty_anneal_iters

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    
    stime = time.time()

    
    criterion = nn.CrossEntropyLoss(reduction='mean')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim
    
    steps = n_steps
    epochs = args.epochs
    if linear_probing:
      lin_epochs = args.lin_epochs
      for param in model.bert.parameters():
          param.requires_grad = False
    
    dataloaders = [envs[i]["train_dataloader"] for i in range(len(envs)) if envs[i]['train']==True]
    
    d_num = len(dataloaders)
    
    val_avg_acc_ls = []
    val_min_acc_ls = []
    val_loss_ls = []
    train_acc_ls = []
    train_loss_ls = [] 

    print("number of envs: ", d_num)
    best_model_val = 0
    best_model_config = {}

    
    model.train()

    for epoch in range(epochs):

        if linear_probing:
            if epoch == lin_epochs:
                for param in model.bert.parameters():
                    param.requires_grad = True

                print("Finished Probing, Reset Optimizer")
                optimizer = Adam(model.parameters(), lr = args.learning_rate)

        iters = [iter(d) for d in dataloaders]
        
        train_acc_ls_epoch = []
        train_loss_ls_epoch = []


        if(epoch == 0):
            print("Training Begins")    
            print("Method: ", args.method)
            print("Information Bottleneck Penalty: ", args.ib_lambda > 0)
            #print("Batch size: ", args.batch_size)
        
        if((epoch+1)%args.epoch_print_step == 0):
            print("Epoch: ", epoch, ", Number of Iterations: ", steps)    
        
        for step in range(steps):

            for i in range(d_num):
                
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

                ### save memory
                train_label.to("cpu")
                mask.to("cpu")
                input_id.to("cpu")  


            train_loss = torch.stack([envs[i]['loss'] for i in range(d_num) if envs[i]['train']==True]).mean()
            train_loss_ls_epoch.append(train_loss.item())

            train_acc = torch.stack([envs[i]['acc'] for i in range(d_num) if envs[i]['train']==True]).mean()

            train_acc_ls_epoch.append(train_acc.item())

        
            weight_norm = torch.tensor(0.).cuda()
            for w in model.parameters():
                weight_norm += w.norm().pow(2)

            train_loss+=(l2_regularizer_weight*weight_norm)

            ### error

            if(args.ib_lambda > 0.):

                ib_weight = args.ib_lambda if step >= args.ib_step else 0.
                var_loss = torch.stack([envs[i]['loss'] for i in range(d_num) if envs[i]['train']==True]).mean()
                train_loss += ib_weight * var_loss

            if method == 'irm':
                
                train_penalty = torch.stack([envs[i]['penalty'] for i in range(d_num) \
                                            if (envs[i]['train']==True and envs[i]["penalty_condition"]==True)]).mean()
                penalty_weight = (p_weight if epoch >= penalty_anneal_iters else .0)
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

            if (step+1) % (steps//3) == 0 and (epoch+1)%args.epoch_print_step == 0:
                print(f'Steps: {step+1} | Training Loss: {sum(train_loss_ls_epoch)/len(train_loss_ls_epoch):.3f} | \
                            Training Accuracy: {sum(train_acc_ls_epoch)/len(train_acc_ls_epoch):.3f}')

        
        
        epoch_train_loss = sum(train_loss_ls_epoch)/len(train_loss_ls_epoch)
        epoch_train_acc = sum(train_acc_ls_epoch)/len(train_acc_ls_epoch)
        
        train_loss_ls.append(epoch_train_loss)
        train_acc_ls.append(epoch_train_acc)
        #if (epoch+1)%args.epoch_print_step == 0:
        #    print("###### Epoch Training Accuracy: ", epoch_train_acc)
        

        ### validate model
        epoch_val_acc = []
        epoch_val_loss = []

        val_dataloaders = [envs[i]["val_dataloader"] for i in range(len(envs)) if envs[i]['train']==True]
        
        for val_dataloader in val_dataloaders:
            val_loss, val_acc = evaluate(model, val_dataloader, use_cuda)
            epoch_val_loss.append(val_loss)
            epoch_val_acc.append(val_acc)

        ### avg
        avg_val_acc = np.mean(epoch_val_acc)
        avg_val_loss = np.mean(epoch_val_loss)

        ### worst
        min_val_acc = np.min(epoch_val_acc)

        val_avg_acc_ls.append(avg_val_acc)
        val_min_acc_ls.append(min_val_acc)
        val_loss_ls.append(avg_val_loss)
        

        if ((avg_val_acc) > best_model_val) and (epoch >= epochs/2) and args.save_best_model:
            if method == "irm" and (epoch > penalty_anneal_iters+10):
                print("### save best model ###")
                model.to('cpu')  # moves model (its parameters) to cpu
                best_model_val = avg_val_acc
                #model_path_pt = args.model_path+ '_checkpoint.pt'
                best_model_config = {'epoch': epoch,
                                    'model_state_dict':  copy.deepcopy(model.state_dict()),
                                    'validation_acc': best_model_val}
                PATH = args.model_path+'_best_model_ckpt'
                torch.save(best_model_config, PATH)

                model.to(device)

            elif method == "erm":
                model.to('cpu')  # moves model (its parameters) to cpu
                best_model_val = avg_val_acc
                #model_path_pt = args.model_path+ '_checkpoint.pt'
                best_model_config = {'epoch': epoch,
                                    'model_state_dict':  copy.deepcopy(model.state_dict()),
                                    'validation_acc': best_model_val}
                
                PATH = args.model_path+'_best_model_ckpt'
                torch.save(best_model_config, PATH)

                model.to(device)


        model.train()

        
        if (epoch+1)%args.epoch_print_step == 0:
            print(f'Epoch: {epoch} | Training Loss: {sum(train_loss_ls_epoch)/len(train_loss_ls_epoch):.3f} | \
                        Training Accuracy: {sum(train_acc_ls_epoch)/len(train_acc_ls_epoch):.3f}')
            #for i, val_acc_i in enumerate(epoch_val_acc):
            print(f'Validation Accuracy: {avg_val_acc:.3f} ')
        
    
    # if args.save_best_model == True:
    # save last model
    last_model_config = {'epoch': epoch,
                        'model_state_dict':  copy.deepcopy(model.state_dict()),
                        'validation_acc': avg_val_acc}
    PATH = args.model_path+'_last_model_ckpt'
    torch.save(last_model_config, PATH)
        ### load 

        #checkpoint = torch.load(PATH)
        #model.load_state_dict(checkpoint['model_state_dict'])
        #epoch = checkpoint['epoch']
        #val_accuracy = checkpoint['validation_acc']
    



    ### log_results_in_dataframe

    if args.save_training_history:
        df = pd.DataFrame(list(zip([i+1 for i in range(epochs)], train_loss_ls, train_acc_ls, val_avg_acc_ls, val_min_acc_ls, val_loss_ls)),\
                                columns =['epochs', 'train_loss', 'train_acc', 'val_avg_acc', 'val_min_acc', 'val_loss'])
        df.to_pickle(args.model_path+'_training_history')

    if(epochs == 0):
        return 0, 0, 0
    return train_acc_ls[-1], val_avg_acc_ls[-1], val_min_acc_ls[-1]
        


def eval_distance(model, envs, batch_size=4, class_wise = True, class_labels = [0, 1, 2, 3, 4, 5], output_layer = 2):

    dataloaders = [envs[i]["dataloader"] for i in range(len(envs))]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    d_num = len(dataloaders)
    iters = [iter(d) for d in dataloaders]
    
    print("number of envs: ", d_num)
    
    model.evaluate()

    ### logit clusters
    representations_logits = [(np.zeros((len(envs[i]["dataloader"]), 6)), np.zeros((len(envs[i]["dataloader"])))) for i in range(d_num)]

    ### inter_logits
    representations_hidden = [np.zeros((len(envs[i]["dataloader"]), 768)) for i in range(d_num)]


    for i in range(d_num):
        
        steps = len(envs[i]["dataloader"])//batch_size
        counter=0
        for j in range(steps):  
        
            current_batch = next(iters[i])

            train_label_0 = current_batch[1]
            train_input_0 = current_batch[0]
            train_label = train_label_0.to(device)
            mask = train_input_0['attention_mask'].to(device)
            input_id = train_input_0['input_ids'].squeeze(1).to(device)

            if output_layer != 0:
                logits, inter_logits = model(input_id, mask, output_layer)
                batch_len = train_input_0.size(0)

                representations_logits[i][0][counter: counter+batch_len] += logits.cpu().detach().numpy()
                representations_logits[i][1][counter: counter+batch_len] += train_label.cpu().detach().numpy()  

                representations_hidden[i][0][counter: counter+batch_len] += inter_logits.cpu().detach().numpy() 

            else:
                logits = model(input_id, mask)
                representations_logits[i][0][counter: counter+batch_len] += logits.cpu().detach().numpy()
                representations_logits[i][1][counter: counter+batch_len] += train_label.cpu().detach().numpy()  


        #env['nll'] = mean_nll(logits, env['labels'][val_size:])
        

    return representations_logits, representations_hidden


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='time alignment -- BERT IRM')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory')
    parser.add_argument('--training_years', nargs='+', default = ['1980','1990','2000','2005','2010'], help='train list of partitioned periods, indicated by the starting year')
    parser.add_argument('--testing_years', nargs='+', default = ['1980','1990','2000','2005','2010'], help='test list of partitioned periods, indicated by the starting year')
    parser.add_argument('--train_conditioning', nargs='+', default = ['1980','1990','2000','2005','2010'], help='environments for conditioning')
    parser.add_argument('--output_file',  default = 'data/output_erm', help='output file to save results')
    parser.add_argument('--method',  type = str, default = "erm" , help='ERM or IRM')
    parser.add_argument('--epochs',  type = int, default = 50, help='number of training epochs')
    parser.add_argument('--learning_rate',  type = float, default =1e-6 , help='learning rate')
    parser.add_argument('--l2_regularizer',  type = float, default =0.00 , help='l2_regularizer_weight for IRM')
    parser.add_argument('--penalty_weight',  type = float, default = 1e3 , help='penalty_weight for IRM')
    parser.add_argument('--penalty_anneal_iters',  type = float, default=10 , help='penalty_anneal_iters for IRM')
    parser.add_argument('--inter',  type = int, default = 2 , help='specify layer for imposing bottleneck penalty')
    parser.add_argument('--ib_lambda',  type = float, default = 0.0 , help='IB penalty weight')
    parser.add_argument('--class_condition',  type = float, default = False , help='IB penalty classwise application')
    parser.add_argument('--ib_step',  type = int, default = 10 , help='penalty_anneal_iters for IB')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--model_path', type=str, default='saved_models/', help='checkpoints')
    parser.add_argument('--data_split', type=str, default='equal_split', help='define env split')
    parser.add_argument('--model', type=str, default='bert', help='language model for finetuning, bert or distilbert')
    parser.add_argument('--linear_probing', type=bool, default=False, help='do linear finetuning')
    parser.add_argument('--lin_epochs', type=int, default=20, help='epochs for linear finetuning')
    parser.add_argument('--seed', type=int, default=0, help='seed for model init')
    parser.add_argument('--save_training_history', type=bool, default=True, help='save stats in dataframe')
    parser.add_argument('--save_best_model', type=bool, default=True, help='save best model weights based on avg validation acc')
    parser.add_argument('--epoch_print_step', type=int, default=1, help='epochs for printing model performance')
    parser.add_argument('--task', type=str, default='scierc', help='select nlp tasks: scierc, aic')
    
    args = parser.parse_args()
    
    training_years = args.training_years
    testing_years = args.testing_years
    train_conditioning = args.train_conditioning
    data_dir = args.data_dir
    output_file = args.output_file
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    model_name = args.model
    data_split = args.data_split
    linear_probing = args.linear_probing
    seed = args.seed
    nlp_task = args.task
    print("Args penalty_anneal_iters: ", args.penalty_anneal_iters)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("conditioned on years: ", train_conditioning)
    
    
    stime = time.time()
    files = []
    val_files = []
    test_res = []
    dataloaders = []
    
    criterion = nn.CrossEntropyLoss()


    random.seed(seed)
    torch.manual_seed(seed)

    #### read data
    
    training_data = []
    training_label = []
    lengths = []
    val_files = []

    if nlp_task == 'scierc':
        print('Dataset: SciERC')
        num_classes = 6
    elif nlp_task == 'aic':
        print('Dataset: AIC')
        num_classes = 2

    train_period = '_'.join(training_years)
    if model_name == 'bert':
        model = BertClassifier(num_classes=num_classes)
        tokenizer_name = 'bert-base-cased'
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    elif model_name == 'distilbert':
        model = DistilBertClassifier()
        tokenizer_name = 'distilbert-base-cased'
        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
    elif model_name == 'gpt2':
        model = GPT2Classifier()
        tokenizer_name = 'gpt2'
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise NotImplementedError
    print("tokenizer used: ", tokenizer_name)
    
    
    # model = model.cuda()
    optimizer = Adam(model.parameters(), lr= learning_rate, eps=1e-08)
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
        train = Dataset(train_data, tokenizer,nlp_task)
        train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        envs[i]["train_dataloader"] = train_dataloader
        
        envs[i]["train"] = True
        envs[i]["year"] = training_years[i]
        if(envs[i]["year"] in train_conditioning):
            envs[i]["penalty_condition"] = True
        else:
            envs[i]["penalty_condition"] = False

        i+=1
    #### add validation environments here
    
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
        val_data = Dataset(val_data, tokenizer,nlp_task)
    
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
        envs[i]["val_dataloader"] = val_dataloader
        i+=1

    final_train_acc, final_avg_valid_acc, final_worst_grp_valid_acc = \
                            train_model(steps, envs, model, optimizer, args, args.method, linear_probing)
                            
    print("Final Model: Validation (Avg, Worst)", final_avg_valid_acc, final_worst_grp_valid_acc)

    ### if load best model, uncomment below

    #PATH = args.model_dir+'_best_model_ckpt'
    #checkpoint = torch.load(PATH)
    #model.load_state_dict(checkpoint['model_state_dict'])
    #best_epoch = checkpoint['epoch']
    #best_val_accuracy = checkpoint['validation_acc']
    #model.to(device)
    #use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda" if use_cuda else "cpu")

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



    ### get final testing accuracy

    testing_accs = []
    test_years = []

    for yr_test in testing_years:
        g = glob.glob("{}/{}/test/{}*".format(data_dir, data_split, yr_test))


        if(yr_test not in training_years):
            ### add more data from years not trained on (extra OOD testing data)
            g1 = glob.glob("{}/{}/train/{}*".format(data_dir, data_split, yr_test))
        
        else:
            g1 = None

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
        
        ### uncomment if extra ood testing data is needed
        # if g1 is not None:
        #     with open(g1[0], 'r') as f:
        #         lines = f.readlines()
        #         for line in lines:
        #             d = json.loads(line)
        #             test_text_list.append(d["text"])
        #             test_labels_list.append(d["labels"])

        
        test_data = [test_text_list, test_labels_list]

        test_data = Dataset(test_data, tokenizer,nlp_task)

        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
        
        test_loss, test_acc = evaluate(model, test_dataloader, use_cuda)

        testing_accs.append(test_acc) 

        d = {"test_period":yr_test,"test_acc":test_acc}
        print(d)
        test_years.append(yr_test)
    
    # all periods tested together (overall accuracy)
    if len(testing_years) > 1:
        #test_data = [all_test_text, all_test_label]
        print("all testing periods")

        d = {"test_period":','.join(testing_years), "test_acc":np.mean(testing_accs)}
        print(d)
    

    df = pd.DataFrame(list(zip(test_years, testing_accs)),\
                                columns =['test_year', 'test_avg_acc'])
    df.to_pickle(args.model_path+'_model_test_results')

    model.to("cpu")
    
    etime=time.time()
    print("time: ", (etime-stime)/60)