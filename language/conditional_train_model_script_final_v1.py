import pandas as pd
import torch
import numpy as np
from models import *
from training_utils import *
from data_reader import prep_scierc, prep_aic
from torch import nn
from torch.optim import Adam
import json
import glob
from torch import autograd
import time
import argparse
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='time alignment -- BERT IRM')
    #parser.add_argument('--data_dir', type=str, default='data', help='data directory')
    parser.add_argument('--raw_data', type=str, default='data', help='data directory')
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
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--model_path', type=str, default='saved_models/', help='checkpoints')
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
    raw_data = args.raw_data
    output_file = args.output_file
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    model_name = args.model
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
        data_dir = prep_scierc(raw_data)
    elif nlp_task == 'aic':
        print('Dataset: AIC')
        num_classes = 2
        data_dir = prep_aic(raw_data)

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
        
        #g = glob.glob("{}/{}/train/{}*".format(data_dir, data_split, yr))
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
            

        #g_val = glob.glob("{}/{}/val/{}*".format(data_dir, data_split, yr))
        g_val = glob.glob("{}/val/{}*".format(data_dir, yr))
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
        #g = glob.glob("{}/{}/test/{}*".format(data_dir, data_split, yr_test))
        g = glob.glob("{}/test/{}*".format(data_dir, yr_test))


        if(yr_test not in training_years):
            ### add more data from years not trained on (extra OOD testing data)
            #g1 = glob.glob("{}/{}/train/{}*".format(data_dir, data_split, yr_test))
            g1 = glob.glob("{}/train/{}*".format(data_dir, yr_test))
        
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