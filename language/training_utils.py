import pandas as pd
import torch
import numpy as np
#from torch._C import int8
from torch import nn
from torch.optim import Adam
from torch import autograd
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
    
def evaluate(model, val_loader, use_cuda):
    
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
    
    #stime = time.time()
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
        
        if((epoch+1)%args.epoch_print_step == 0):
            print("Epoch: ", epoch)    
        
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
                train_penalty = envs[0]['penalty'] * 0.  # so that this term is a tensor
            else:
                raise NotImplementedError    

            model.zero_grad()
            train_loss.backward()
            optimizer.step()

            # if (step+1) % (steps//3) == 0 and (epoch+1)%args.epoch_print_step == 0:
            #     print(f'Steps: {step+1} | Training Loss: {sum(train_loss_ls_epoch)/len(train_loss_ls_epoch):.3f} | \
            #                 Training Accuracy: {sum(train_acc_ls_epoch)/len(train_acc_ls_epoch):.3f}')

        epoch_train_loss = sum(train_loss_ls_epoch)/len(train_loss_ls_epoch)
        epoch_train_acc = sum(train_acc_ls_epoch)/len(train_acc_ls_epoch)
        
        train_loss_ls.append(epoch_train_loss)
        train_acc_ls.append(epoch_train_acc)

        ### validate model at each epoch
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

    ### log_results_in_dataframe if needed
    if args.save_training_history:
        df = pd.DataFrame(list(zip([i+1 for i in range(epochs)], train_loss_ls, train_acc_ls, val_avg_acc_ls, val_min_acc_ls, val_loss_ls)),\
                                columns =['epochs', 'train_loss', 'train_acc', 'val_avg_acc', 'val_min_acc', 'val_loss'])
        df.to_pickle(args.model_path+'_training_history')

    if(epochs == 0):
        return 0, 0, 0
    return train_acc_ls[-1], val_avg_acc_ls[-1], val_min_acc_ls[-1]
