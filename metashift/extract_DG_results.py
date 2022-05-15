import pandas as pd
import numpy as np
import json
import argparse
import glob
import re
import pickle

def find_param(param_name, gfile):
        s1,s2 = re.search(param_name,gfile).span()
        p1 = gfile[s2:]
        s1, s2 = re.search('_',p1).span()
        val = p1[:s1]
        return val

def extract_results_irm(minoririty_groups, exp_id, envs,seeds, anneals, penals, lrs, bss):
    res_irm = []
    pen_df = {
        f'{envs[0]}_p0_acc_val':[],
            f'{envs[0]}_p1_acc_val':[],
            f'{envs[0]}_p0_auc_val':[],
            f'{envs[0]}_p1_auc_val':[],
            f'{envs[0]}_p0_acc_test':[],
            f'{envs[0]}_p1_acc_test':[],
            f'{envs[0]}_p0_auc_test':[],
            f'{envs[0]}_p1_auc_test':[],

            f'{envs[1]}_p0_acc_val':[],
            f'{envs[1]}_p1_acc_val':[],
            f'{envs[1]}_p0_auc_val':[],
            f'{envs[1]}_p1_auc_val':[],
            f'{envs[1]}_p0_acc_test':[],
            f'{envs[1]}_p1_acc_test':[],
            f'{envs[1]}_p0_auc_test':[],
            f'{envs[1]}_p1_auc_test':[],
            
            f'{envs[0]}_avg_acc_val':[],
            f'{envs[0]}_avg_auc_val':[],
        
            f'{envs[1]}_avg_acc_val':[],
            f'{envs[1]}_avg_auc_val':[],
        
            f'{envs[0]}_avg_acc_test':[],
            f'{envs[0]}_avg_auc_test':[],
        
            f'{envs[1]}_avg_acc_test':[],
            f'{envs[1]}_avg_auc_test':[],

            'avg_minorirty_acc_val' : [],
            'avg_minorirty_auc_val' : [],
            'avg_minorirty_acc_test' : [],
            'avg_minorirty_auc_test': [],

            'avg_acc_val' : [],
            'avg_auc_val' : [],
            'avg_acc_test' : [],
            'avg_auc_test': []
            }
    # seeds = []
    # lrs = []
    # penals = []
    # anneals = []
    # bss = []
    
    # for gfile in glob.glob(f'results/{exp_id}_irm/*'):
    #     s = find_param('_seed_',gfile)
    #     seeds.append(int(s))
        
    #     a = find_param('_anneal_',gfile)
    #     anneals.append(float(a))
        
    #     p = find_param('_pen_', gfile)
    #     penals.append(float(p))
        
    #     lr = find_param('_lr_',gfile)
    #     lrs.append(float(lr))
        
    #     bs = find_param('_bs_',gfile)
    #     bss.append(int(bs))
    # seeds = sorted(set(seeds))
    # lrs = sorted(set(lrs))
    # penals = sorted(set(penals))
    # anneals = sorted(set(anneals))
    # bss = sorted(set(bss))
    for batch_size in bss:
        for anneal in anneals:
            for lr in lrs:
                for pen in penals:
                    for i, seed in enumerate(seeds):
                        mino_groups = minoririty_groups[i]
                        # print(mino_groups)
                        minority_acc_val = 0
                        minority_auc_val = 0
                        minority_acc_test = 0
                        minority_auc_test = 0


                        acc_val = 0
                        auc_val = 0
                        acc_test = 0
                        auc_test = 0

                        counter = 0


                        min_groups = [f"{envs[0]}_{mino_groups[0]}", f"{envs[1]}_{mino_groups[1]}"]

#                         print(exp_partition)
                        gfile = f"results/{exp_id}_irm/{exp_id}_seed_{seed}_pen_{pen}_lr_{lr}_anneal_{anneal}_bs_{batch_size}_partition_val.json"
                        with open(gfile,'r') as f:
                            df = json.load(f)
                        env_acc= {f"{envs[0]}":0,f"{envs[1]}":0}
                        env_auc= {f"{envs[0]}":0,f"{envs[1]}":0}
                        for gr in df:
#                             print(gr)
                            p = gr['groups_local'].split("_")[-1]
                            
                            env_p = gr['groups_local'].split("_")[0]
                            pen_df[f"{gr['groups_local']}_acc_val"].append(float(gr['acc']))
                            pen_df[f"{gr['groups_local']}_auc_val"].append(float(gr['auc']))
                            
                            env_acc[env_p]+=float(gr['acc'])
                            env_auc[env_p]+=float(gr['auc'])
                            for min_gr in min_groups:
                                if gr['groups_local'] == min_gr:
                                    counter+=1
                                    minority_acc_val+=float(gr['acc'])
                                    minority_auc_val+=float(gr['auc'])
                            acc_val+=float(gr['acc'])
                            auc_val+=float(gr['auc'])
                        
                        for env_p in [0,1]:
                            pen_df[f'{envs[env_p]}_avg_acc_val'].append(env_acc[envs[env_p]]/2)
                            pen_df[f'{envs[env_p]}_avg_auc_val'].append(env_auc[envs[env_p]]/2)


                        gfile = f"results/{exp_id}_irm/{exp_id}_seed_{seed}_pen_{pen}_lr_{lr}_anneal_{anneal}_bs_{batch_size}_partition_test.json"

                        with open(gfile,'r') as f:
                            df = json.load(f)
                    #                 print(df)
                        
                        env_acc= {f"{envs[0]}":0,f"{envs[1]}":0}
                        env_auc= {f"{envs[0]}":0,f"{envs[1]}":0}

                        for gr in df:
                            p = gr['groups_local'].split("_")[-1]
                            env_p = gr['groups_local'].split("_")[0]
                            
                            pen_df[f"{gr['groups_local']}_acc_test"].append(float(gr['acc']))
                            pen_df[f"{gr['groups_local']}_auc_test"].append(float(gr['auc']))
                            for min_gr in min_groups:
                                if gr['groups_local'] == min_gr:
                                    minority_acc_test+=float(gr['acc'])
                                    minority_auc_test+=float(gr['auc'])
                            acc_test+=float(gr['acc'])
                            auc_test+=float(gr['auc'])
                            
                            env_acc[env_p]+=float(gr['acc'])
                            env_auc[env_p]+=float(gr['auc'])
                        
                        for p in [0,1]:
                            pen_df[f'{envs[p]}_avg_acc_test'].append(env_acc[envs[p]]/2)
                            pen_df[f'{envs[p]}_avg_auc_test'].append(env_auc[envs[p]]/2)

                        pen_df['avg_minorirty_acc_val'].append(minority_acc_val/2)
                        pen_df['avg_minorirty_auc_val'].append(minority_auc_val/2)
                        pen_df['avg_minorirty_acc_test'].append(minority_acc_test/2)
                        pen_df['avg_minorirty_auc_test'].append(minority_auc_test/2)


                        pen_df['avg_acc_val'].append(acc_val/4)
                        pen_df['avg_auc_val'].append(auc_val/4)
                        pen_df['avg_acc_test'].append(acc_test/4)
                        pen_df['avg_auc_test'].append(auc_test/4)



                    d = {k:f'{np.mean(v):.3f}({np.std(v):.2f})' for k,v in pen_df.items()}
            #         d = {k:f'{np.mean(v):.3f}' for k,v in pen_df.items()}
                    new_d = {}
                    new_d['algorithm'] = 'IRM'
                    new_d['penalty'] = pen
                    new_d['lr'] = lr
                    new_d['bs'] = batch_size
                    new_d['anneal'] = anneal
                    new_d.update(d)
                    res_irm.append(new_d)
    return res_irm
def extract_results_mean(minoririty_groups,exp_id, envs,seeds, anneals, penals, lrs, bss):
#     envs = ['p1','p2']
    res_irm = []
    pen_df = {
        f'{envs[0]}_p0_acc_val':[],
            f'{envs[0]}_p1_acc_val':[],
            f'{envs[0]}_p0_auc_val':[],
            f'{envs[0]}_p1_auc_val':[],
            f'{envs[0]}_p0_acc_test':[],
            f'{envs[0]}_p1_acc_test':[],
            f'{envs[0]}_p0_auc_test':[],
            f'{envs[0]}_p1_auc_test':[],

            f'{envs[1]}_p0_acc_val':[],
            f'{envs[1]}_p1_acc_val':[],
            f'{envs[1]}_p0_auc_val':[],
            f'{envs[1]}_p1_auc_val':[],
            f'{envs[1]}_p0_acc_test':[],
            f'{envs[1]}_p1_acc_test':[],
            f'{envs[1]}_p0_auc_test':[],
            f'{envs[1]}_p1_auc_test':[],
            
            f'{envs[0]}_avg_acc_val':[],
            f'{envs[0]}_avg_auc_val':[],
        
            f'{envs[1]}_avg_acc_val':[],
            f'{envs[1]}_avg_auc_val':[],
        
            f'{envs[0]}_avg_acc_test':[],
            f'{envs[0]}_avg_auc_test':[],
        
            f'{envs[1]}_avg_acc_test':[],
            f'{envs[1]}_avg_auc_test':[],

            'avg_minorirty_acc_val' : [],
            'avg_minorirty_auc_val' : [],
            'avg_minorirty_acc_test' : [],
            'avg_minorirty_auc_test': [],

            'avg_acc_val' : [],
            'avg_auc_val' : [],
            'avg_acc_test' : [],
            'avg_auc_test': []
            }
    

    print("anneals:", anneals)
    
#     print(gfile.split("/")[-1].split('_'))
    for batch_size in bss:
        for anneal in anneals:
            for lr in lrs:
                for pen in penals:
                    for i, seed in enumerate(seeds):
#                         exp_id = 'D_ii'
                    #     exp_partition =
                        mino_groups = minoririty_groups[i]
                        minority_acc_val = 0
                        minority_auc_val = 0
                        minority_acc_test = 0
                        minority_auc_test = 0


                        acc_val = 0
                        auc_val = 0
                        acc_test = 0
                        auc_test = 0

                        for i, exp_partition in enumerate(envs):
                            min_group = mino_groups[i]

                            # print(exp_partition)

                            gfile = f"results/{exp_id}_{exp_partition}/{exp_id}_seed_{seed}_pen_{pen}_lr_{lr}_anneal_{anneal}_bs_{batch_size}_partition_val.json"
                            
                            with open(gfile,'r') as f:
                                df = json.load(f)
                            #                 print(df)
                            p_avg_acc = 0
                            p_avg_auc = 0
                            
                            for gr in df:
                                pen_df[f"{exp_partition}_{gr['groups_local']}_acc_val"].append(float(gr['acc']))
                                pen_df[f"{exp_partition}_{gr['groups_local']}_auc_val"].append(float(gr['auc']))
                                if gr['groups_local'] == min_group:
                                    minority_acc_val+=float(gr['acc'])
                                    minority_auc_val+=float(gr['auc'])
                                acc_val+=float(gr['acc'])
                                auc_val+=float(gr['auc'])
                                
                                p_avg_acc+=float(gr['acc'])
                                p_avg_auc+=float(gr['auc'])

                            pen_df[f"{exp_partition}_avg_acc_val"].append(p_avg_acc/2)
                            pen_df[f"{exp_partition}_avg_auc_val"].append(p_avg_auc/2)
                            
                            gfile = f"results/{exp_id}_{exp_partition}/{exp_id}_seed_{seed}_pen_{pen}_lr_{lr}_anneal_{anneal}_bs_{batch_size}_partition_test.json"

                            with open(gfile,'r') as f:
                                df = json.load(f)

                            #                 print(df)
                            p_avg_acc = 0
                            p_avg_auc = 0
                            
                            for gr in df:
                                pen_df[f"{exp_partition}_{gr['groups_local']}_acc_test"].append(float(gr['acc']))
                                pen_df[f"{exp_partition}_{gr['groups_local']}_auc_test"].append(float(gr['auc']))
                                if gr['groups_local'] == min_group:
                                    minority_acc_test+=float(gr['acc'])
                                    minority_auc_test+=float(gr['auc'])
                                acc_test+=float(gr['acc'])
                                auc_test+=float(gr['auc'])
                                
                                p_avg_acc+=float(gr['acc'])
                                p_avg_auc+=float(gr['auc'])
                            pen_df[f"{exp_partition}_avg_acc_test"].append(p_avg_acc/2)
                            pen_df[f"{exp_partition}_avg_auc_test"].append(p_avg_auc/2)

                        pen_df['avg_minorirty_acc_val'].append(minority_acc_val/2)
                        pen_df['avg_minorirty_auc_val'].append(minority_auc_val/2)
                        pen_df['avg_minorirty_acc_test'].append(minority_acc_test/2)
                        pen_df['avg_minorirty_auc_test'].append(minority_auc_test/2)


                        pen_df['avg_acc_val'].append(acc_val/4)
                        pen_df['avg_auc_val'].append(auc_val/4)
                        pen_df['avg_acc_test'].append(acc_test/4)
                        pen_df['avg_auc_test'].append(auc_test/4)



                    d = {k:f'{np.mean(v):.3f}({np.std(v):.2f})' for k,v in pen_df.items()}
            #         d = {k:f'{np.mean(v):.3f}' for k,v in pen_df.items()}
                    new_d = {}
                    new_d['algorithm'] = 'P-IRM'
                    new_d['penalty'] = pen
                    new_d['lr'] = lr
                    new_d['bs'] = batch_size
                    new_d['anneal'] = anneal
                    new_d.update(d)
                    res_irm.append(new_d)
    return res_irm


def extract_results_irm_opt(minoririty_groups, exp_id, envs):
    res_irm = []
    pen_df = {
        f'{envs[0]}_p0_acc_val':[],
            f'{envs[0]}_p1_acc_val':[],
            f'{envs[0]}_p0_auc_val':[],
            f'{envs[0]}_p1_auc_val':[],
            f'{envs[0]}_p0_acc_test':[],
            f'{envs[0]}_p1_acc_test':[],
            f'{envs[0]}_p0_auc_test':[],
            f'{envs[0]}_p1_auc_test':[],

            f'{envs[1]}_p0_acc_val':[],
            f'{envs[1]}_p1_acc_val':[],
            f'{envs[1]}_p0_auc_val':[],
            f'{envs[1]}_p1_auc_val':[],
            f'{envs[1]}_p0_acc_test':[],
            f'{envs[1]}_p1_acc_test':[],
            f'{envs[1]}_p0_auc_test':[],
            f'{envs[1]}_p1_auc_test':[],
            
            f'{envs[0]}_avg_acc_val':[],
            f'{envs[0]}_avg_auc_val':[],
        
            f'{envs[1]}_avg_acc_val':[],
            f'{envs[1]}_avg_auc_val':[],
        
            f'{envs[0]}_avg_acc_test':[],
            f'{envs[0]}_avg_auc_test':[],
        
            f'{envs[1]}_avg_acc_test':[],
            f'{envs[1]}_avg_auc_test':[],

            'avg_minorirty_acc_val' : [],
            'avg_minorirty_auc_val' : [],
            'avg_minorirty_acc_test' : [],
            'avg_minorirty_auc_test': [],

            'avg_acc_val' : [],
            'avg_auc_val' : [],
            'avg_acc_test' : [],
            'avg_auc_test': []
            }
    seeds = []
    
    for gfile in glob.glob(f'results/{exp_id}_irm/*'):
        s = find_param('_seed_',gfile)
        seeds.append(int(s))
        
    seeds = sorted(set(seeds))
   
    for i, seed in enumerate(seeds):
#                         exp_id = 'D_ii'
    #     exp_partition =
        mino_groups = minoririty_groups[i]
        minority_acc_val = 0
        minority_auc_val = 0
        minority_acc_test = 0
        minority_auc_test = 0


        acc_val = 0
        auc_val = 0
        acc_test = 0
        auc_test = 0

        counter = 0


        min_groups = [f"{envs[0]}_{mino_groups[0]}", f"{envs[1]}_{mino_groups[1]}"]

#                         print(exp_partition)
        gfile = glob.glob(f"results/{exp_id}_irm/{exp_id}_seed_{seed}_*_partition_val.json")[0]
        with open(gfile,'r') as f:
            df = json.load(f)
        env_acc= {f"{envs[0]}":0,f"{envs[1]}":0}
        env_auc= {f"{envs[0]}":0,f"{envs[1]}":0}
        for gr in df:
#                             print(gr)
            p = gr['groups_local'].split("_")[-1]

            env_p = gr['groups_local'].split("_")[0]
            pen_df[f"{gr['groups_local']}_acc_val"].append(float(gr['acc']))
            pen_df[f"{gr['groups_local']}_auc_val"].append(float(gr['auc']))

            env_acc[env_p]+=float(gr['acc'])
            env_auc[env_p]+=float(gr['auc'])
            for min_gr in min_groups:
                if gr['groups_local'] == min_gr:
                    counter+=1
                    minority_acc_val+=float(gr['acc'])
                    minority_auc_val+=float(gr['auc'])
            acc_val+=float(gr['acc'])
            auc_val+=float(gr['auc'])

        for env_p in [0,1]:
            pen_df[f'{envs[env_p]}_avg_acc_val'].append(env_acc[envs[env_p]]/2)
            pen_df[f'{envs[env_p]}_avg_auc_val'].append(env_auc[envs[env_p]]/2)


        gfile = glob.glob(f"results/{exp_id}_irm/{exp_id}_seed_{seed}*_partition_test.json")[0]

        with open(gfile,'r') as f:
            df = json.load(f)
    #                 print(df)

        env_acc= {f"{envs[0]}":0,f"{envs[1]}":0}
        env_auc= {f"{envs[0]}":0,f"{envs[1]}":0}

        for gr in df:
            p = gr['groups_local'].split("_")[-1]
            env_p = gr['groups_local'].split("_")[0]

            pen_df[f"{gr['groups_local']}_acc_test"].append(float(gr['acc']))
            pen_df[f"{gr['groups_local']}_auc_test"].append(float(gr['auc']))
            for min_gr in min_groups:
                if gr['groups_local'] == min_gr:
                    minority_acc_test+=float(gr['acc'])
                    minority_auc_test+=float(gr['auc'])
            acc_test+=float(gr['acc'])
            auc_test+=float(gr['auc'])

            env_acc[env_p]+=float(gr['acc'])
            env_auc[env_p]+=float(gr['auc'])

        for p in [0,1]:
            pen_df[f'{envs[p]}_avg_acc_test'].append(env_acc[envs[p]]/2)
            pen_df[f'{envs[p]}_avg_auc_test'].append(env_auc[envs[p]]/2)

        pen_df['avg_minorirty_acc_val'].append(minority_acc_val/2)
        pen_df['avg_minorirty_auc_val'].append(minority_auc_val/2)
        pen_df['avg_minorirty_acc_test'].append(minority_acc_test/2)
        pen_df['avg_minorirty_auc_test'].append(minority_auc_test/2)


        pen_df['avg_acc_val'].append(acc_val/4)
        pen_df['avg_auc_val'].append(auc_val/4)
        pen_df['avg_acc_test'].append(acc_test/4)
        pen_df['avg_auc_test'].append(auc_test/4)



    d = {k:f'{np.mean(v):.3f}({np.std(v):.2f})' for k,v in pen_df.items()}
#         d = {k:f'{np.mean(v):.3f}' for k,v in pen_df.items()}
    new_d = {}
    new_d['algorithm'] = 'IRM'
#     new_d['penalty'] = pen
#     new_d['lr'] = lr
#     new_d['bs'] = batch_size
#     new_d['anneal'] = anneal
    new_d.update(d)
    res_irm.append(new_d)
    return res_irm

def extract_results_mean_opt(minoririty_groups,exp_id, envs):
#     envs = ['p1','p2']
    res_irm = []
    pen_df = {
        f'{envs[0]}_p0_acc_val':[],
            f'{envs[0]}_p1_acc_val':[],
            f'{envs[0]}_p0_auc_val':[],
            f'{envs[0]}_p1_auc_val':[],
            f'{envs[0]}_p0_acc_test':[],
            f'{envs[0]}_p1_acc_test':[],
            f'{envs[0]}_p0_auc_test':[],
            f'{envs[0]}_p1_auc_test':[],

            f'{envs[1]}_p0_acc_val':[],
            f'{envs[1]}_p1_acc_val':[],
            f'{envs[1]}_p0_auc_val':[],
            f'{envs[1]}_p1_auc_val':[],
            f'{envs[1]}_p0_acc_test':[],
            f'{envs[1]}_p1_acc_test':[],
            f'{envs[1]}_p0_auc_test':[],
            f'{envs[1]}_p1_auc_test':[],
            
            f'{envs[0]}_avg_acc_val':[],
            f'{envs[0]}_avg_auc_val':[],
        
            f'{envs[1]}_avg_acc_val':[],
            f'{envs[1]}_avg_auc_val':[],
        
            f'{envs[0]}_avg_acc_test':[],
            f'{envs[0]}_avg_auc_test':[],
        
            f'{envs[1]}_avg_acc_test':[],
            f'{envs[1]}_avg_auc_test':[],

            'avg_minorirty_acc_val' : [],
            'avg_minorirty_auc_val' : [],
            'avg_minorirty_acc_test' : [],
            'avg_minorirty_auc_test': [],

            'avg_acc_val' : [],
            'avg_auc_val' : [],
            'avg_acc_test' : [],
            'avg_auc_test': []
            }
    seeds = []
    
    for gfile in glob.glob(f'results/{exp_id}_p1/*'):
        s = find_param('_seed_',gfile)
        seeds.append(int(s))
        
    seeds = sorted(set(seeds))
   
    for i, seed in enumerate(seeds):
        mino_groups = minoririty_groups[i]
        minority_acc_val = 0
        minority_auc_val = 0
        minority_acc_test = 0
        minority_auc_test = 0


        acc_val = 0
        auc_val = 0
        acc_test = 0
        auc_test = 0
        for i, exp_partition in enumerate(envs):
            min_group = mino_groups[i]
            gfile = glob.glob(f"results/{exp_id}_{exp_partition}/{exp_id}_seed_{seed}*_partition_val.json")[0]

            with open(gfile,'r') as f:
                df = json.load(f)
            #                 print(df)
            p_avg_acc = 0
            p_avg_auc = 0

            for gr in df:
                pen_df[f"{exp_partition}_{gr['groups_local']}_acc_val"].append(float(gr['acc']))
                pen_df[f"{exp_partition}_{gr['groups_local']}_auc_val"].append(float(gr['auc']))
                if gr['groups_local'] == min_group:
                    minority_acc_val+=float(gr['acc'])
                    minority_auc_val+=float(gr['auc'])
                acc_val+=float(gr['acc'])
                auc_val+=float(gr['auc'])

                p_avg_acc+=float(gr['acc'])
                p_avg_auc+=float(gr['auc'])

            pen_df[f"{exp_partition}_avg_acc_val"].append(p_avg_acc/2)
            pen_df[f"{exp_partition}_avg_auc_val"].append(p_avg_auc/2)

            gfile = glob.glob(f"results/{exp_id}_{exp_partition}/{exp_id}_seed_{seed}*_partition_test.json")[0]
            with open(gfile,'r') as f:
                df = json.load(f)
            #                 print(df)
            p_avg_acc = 0
            p_avg_auc = 0

            for gr in df:
                pen_df[f"{exp_partition}_{gr['groups_local']}_acc_test"].append(float(gr['acc']))
                pen_df[f"{exp_partition}_{gr['groups_local']}_auc_test"].append(float(gr['auc']))
                if gr['groups_local'] == min_group:
                    minority_acc_test+=float(gr['acc'])
                    minority_auc_test+=float(gr['auc'])
                acc_test+=float(gr['acc'])
                auc_test+=float(gr['auc'])

                p_avg_acc+=float(gr['acc'])
                p_avg_auc+=float(gr['auc'])
            pen_df[f"{exp_partition}_avg_acc_test"].append(p_avg_acc/2)
            pen_df[f"{exp_partition}_avg_auc_test"].append(p_avg_auc/2)

        pen_df['avg_minorirty_acc_val'].append(minority_acc_val/2)
        pen_df['avg_minorirty_auc_val'].append(minority_auc_val/2)
        pen_df['avg_minorirty_acc_test'].append(minority_acc_test/2)
        pen_df['avg_minorirty_auc_test'].append(minority_auc_test/2)


        pen_df['avg_acc_val'].append(acc_val/4)
        pen_df['avg_auc_val'].append(auc_val/4)
        pen_df['avg_acc_test'].append(acc_test/4)
        pen_df['avg_auc_test'].append(auc_test/4)

    d = {k:f'{np.mean(v):.3f}({np.std(v):.2f})' for k,v in pen_df.items()}
#         d = {k:f'{np.mean(v):.3f}' for k,v in pen_df.items()}
    new_d = {}
    new_d['algorithm'] = 'P-IRM'
    new_d.update(d)
    res_irm.append(new_d)
    return res_irm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Results extraction')
    
    parser.add_argument('--results_file_name', type=str, default='results')
    parser.add_argument('--experiment_name', type=str,default='experiment_D')
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--opt', action='store_true',help='indicator if this is not a param search, but a final optimal choice')
    # parser.add_argument('--minority_groups', nargs='+', default = ['p0','p0'], help='minority groups')

    args = parser.parse_args()
    # exp_id = 'experiment_D_50_6'
    exp_id = args.experiment_name
    results_file = args.results_file_name
    
    
    
    seeds = []
    lrs = []
    penals = []
    anneals = []
    bss = []
    
    for gfile in glob.glob(f'results/{exp_id}_p1/*_test*'):
        s = find_param('_seed_',gfile)
        seeds.append(int(s))
        
        a = find_param('_anneal_',gfile)
        anneals.append(float(a))
        
        p = find_param('_pen_', gfile)
        penals.append(float(p))
        
        lr = find_param('_lr_',gfile)
        lrs.append(float(lr))
        
        bs = find_param('_bs_',gfile)
        bss.append(int(bs))

    seeds1 = sorted(set(seeds))
    lrs1 = sorted(set(lrs))
    penals1 = sorted(set(penals))
    anneals1 = sorted(set(anneals))
    bss1 = sorted(set(bss))

    seeds = []
    lrs = []
    penals = []
    anneals = []
    bss = []

    for gfile in glob.glob(f'results/{exp_id}_p2/*_test*'):
        s = find_param('_seed_',gfile)
        seeds.append(int(s))
        
        a = find_param('_anneal_',gfile)
        anneals.append(float(a))
        
        p = find_param('_pen_', gfile)
        penals.append(float(p))
        
        lr = find_param('_lr_',gfile)
        lrs.append(float(lr))
        
        bs = find_param('_bs_',gfile)
        bss.append(int(bs))
    print("## intersection 1")
    seeds = sorted(set(seeds).intersection(set(seeds1)))
    lrs = sorted(set(lrs).intersection(set(lrs1)))
    penals = sorted(set(penals).intersection(set(penals1)))
    anneals = sorted(set(anneals).intersection(set(anneals1)))
    bss = sorted(set(bss).intersection(set(bss1)))

    seeds2 = []
    lrs2 = []
    penals2 = []
    anneals2 = []
    bss2 = []

    for gfile in glob.glob(f'results/{exp_id}_irm/*_test*'):
        s = find_param('_seed_',gfile)
        seeds2.append(int(s))
        
        a = find_param('_anneal_',gfile)
        anneals2.append(float(a))
        
        p = find_param('_pen_', gfile)
        penals2.append(float(p))
        
        lr = find_param('_lr_',gfile)
        lrs2.append(float(lr))
        
        bs = find_param('_bs_',gfile)
        bss2.append(int(bs))
    print("## intersection 2")
    print(set(anneals2))
    print(set(penals2))
    seeds = sorted(set(seeds).intersection(set(seeds2)))
    lrs = sorted(set(lrs).intersection(set(lrs2)))
    penals = sorted(set(penals).intersection(set(penals2)))
    anneals = sorted(set(anneals).intersection(set(anneals2)))
    bss = sorted(set(bss).intersection(set(bss2)))
    print("final anneals",anneals)
    print("final penals",penals)
    
    # anneals = [100.0]
    # penals = [100.0,200.0,500.0,1000.0]
    
    minoririty_groups = []
    gs = glob.glob(f"{args.data_folder}*")
    for g in gs:
        with open(f"{g}/minority_groups.pkl", 'rb') as pf:
            group_dict = pickle.load(pf)
        group_dict = list(group_dict)
        if len(group_dict) == 1:
            group_dict.append(group_dict[0])
        minoririty_groups.append(group_dict)
    print("minorities at each seed: ",minoririty_groups)

    if args.opt == True:
       
        envs = ['p1','p2']
        p1_res = extract_results_mean_opt(minoririty_groups, exp_id, envs)
        d1 = pd.DataFrame(p1_res)

        envs = ['indoor','outdoor']
        
        

        p1_res = extract_results_irm_opt(minoririty_groups, exp_id, envs)
        d2 = pd.DataFrame(p1_res)
        assert len(d2.columns) == len(d1.columns)
        d1 = d1.rename(columns=dict(zip(d1.columns,d2.columns)))
    else:
       
        envs = ['p1','p2']
        p1_res = extract_results_mean(minoririty_groups, exp_id, envs, seeds, anneals, penals, lrs, bss)
        d1 = pd.DataFrame(p1_res)

        envs = ['indoor','outdoor']
        

        p1_res = extract_results_irm(minoririty_groups, exp_id, envs,seeds, anneals, penals, lrs, bss)
        d2 = pd.DataFrame(p1_res)
        assert len(d2.columns) == len(d1.columns)
        d1 = d1.rename(columns=dict(zip(d1.columns,d2.columns)))
    df = pd.concat([d1,d2])
    df.to_csv(results_file)
