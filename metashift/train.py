import sys
import os
import argparse
import numpy as np
import glob
import json

def train_experiment(args, base_script, domains_config, data_indicator):
    seed = args.seed
    print('--seed', seed)
    if args.partial:
        alg_id = f'{data_indicator}{args.algorithm}'
    else:
        alg_id = f'{args.algorithm}'

    script = base_script
    script += ' --domains '+domains_config
    script += f' --experiment {alg_id}'
    script += ' --seed '+str(seed)
    script += f' --output_dir {args.output_dir}_{alg_id}'
    script += f' --data {args.data_dir}/{data_indicator}'
    
    results_dir = f'{args.results_folder}/{args.experiment_id}_{alg_id}'
    os.makedirs(results_dir,exist_ok=True)
    script += ' --results_file '+results_dir

    if args.algorithm == 'irm':
        irm_anneal = args.irm_anneal
        irm_penalty = args.irm_penalty

        script += ' --irm_lambda '+ str(irm_penalty)
        script += ' --anneal '+str(irm_anneal)
        script += ' --algorithm IRM'
    elif args.algorithm == 'ibirm':
        irm_anneal = args.irm_anneal
        irm_penalty = args.irm_penalty
        ibirm_anneal = args.ibirm_anneal
        ibirm_penalty = args.ibirm_penalty

        script += ' --irm_lambda '+ str(irm_penalty)
        script += ' --anneal '+str(irm_anneal)
        script += ' --ib_penalty_anneal_iters '+str(ibirm_anneal)
        script += ' --ib_lambda '+str(ibirm_penalty)
        script += ' --algorithm IB_IRM'
    elif args.algorithm == 'erm':
        script += ' --algorithm ERM'
    else:
        raise Exception('Use only "erm", "irm", "ibirm" in lower case for --algorithm')

    
    os.system(script)

def read_results(args, data_indicator):
    acc_dict = {}
    if args.partial:
        alg_id = f'{data_indicator}{args.algorithm}'
    else:
        alg_id = f'{args.algorithm}'
    if args.algorithm == 'irm':
        irm_anneal = str(float(args.irm_anneal))
        irm_penalty = str(float(args.irm_penalty))
        filename = f'{args.results_folder}/{args.experiment_id}_{alg_id}/{args.experiment_id}_seed_{args.seed}_pen_{irm_penalty}_*_anneal_{irm_anneal}*details_test.json'
    elif args.algorithm == 'ibirm':
        irm_anneal = str(float(args.irm_anneal))
        irm_penalty = str(float(args.irm_penalty))
        ibirm_anneal = str(float(args.ibirm_anneal))
        ibirm_penalty = str(float(args.ibirm_penalty))
        filename = f'{args.results_folder}/{args.experiment_id}_{alg_id}/{args.experiment_id}_seed_{args.seed}_pen_{irm_penalty}_*_anneal_{irm_anneal}_penibirm_{ibirm_penalty}_ibanneal_{ibirm_anneal}_bs_8_partition_details_test.json'

    elif args.algorithm == 'erm':
        filename = f'{args.results_folder}/{args.experiment_id}_{alg_id}/{args.experiment_id}_seed_{args.seed}_*_details_test.json'
    else:
        raise Exception('Use only "erm", "irm", "ibirm" in lower case for --algorithm')

    # know minorities in SP
    if args.experiment == 'SP':
        dog_p1 = len(glob.glob(f'{args.data_dir}/irm/train/dog/dog(P1)/*'))
        dog_p2 = len(glob.glob(f'{args.data_dir}/irm/train/dog/dog(P2)/*'))
        cat_p1 = len(glob.glob(f'{args.data_dir}/irm/train/cat/cat(P1)/*'))
        cat_p2 = len(glob.glob(f'{args.data_dir}/irm/train/cat/cat(P2)/*'))
        ps = ['P1','P2'] 
        min_cat = ps[np.argmin([cat_p1,cat_p2])]
        min_dog = ps[np.argmin([dog_p1,dog_p2])]

    for g in glob.glob(filename):
        with open(g,'r') as f:
            df_all = json.load(f)
    for df in df_all:
        if args.experiment == 'DG':
            print(f"{df['groups_local']} \t {df['acc']} \t\t{df['size']}")
        elif args.experiment == 'SP':
            if (df['groups_local'].split('_')[0] == f'dog({min_dog})') or (df['groups_local'].split('_')[0] == f'cat({min_cat})'):
                print(f"{df['groups_local'].split('_')[0]} \t {df['acc']} \t\t{df['size']} \t (minority)")
            else:
                print(f"{df['groups_local'].split('_')[0]} \t {df['acc']} \t\t{df['size']}")
        else:
            raise Exception('Use only "DG" or "SP" in upper case for --experiment')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genome Dataset')

    parser.add_argument('--irm_penalty', type=int, default = 100, help='irm penalty')
    parser.add_argument('--irm_anneal',  type=int, default = 20)
    parser.add_argument('--ibirm_penalty', type=int, default = 100)
    parser.add_argument('--ibirm_anneal',  type=int, default = 20)

    parser.add_argument('--seed',  type=int, default = 0)
    parser.add_argument('--experiment',type = str, default='DG', help='options: DG for Domain Generalization experiments. SP for Subpopulationshifts experiments')
    parser.add_argument('--experiment_id', type=str,help='a unique experiment name in order to save and extract results', default='cat_dog_exp1_A')

    parser.add_argument('--algorithm', default='irm',help='algorithms either: erm, irm, ibirm')
    parser.add_argument('--partial', action='store_true',help='apply partial invariance p-algorithm e.g. p-irm')
    parser.add_argument('--output_dir',default='train_ouptut')
    parser.add_argument('--data_dir', type=str,help='data directory')
    parser.add_argument('--results_folder', type=str,help='raw results directory', default='results')

    args = parser.parse_args()
    
    partial = args.partial

    base_script = 'python experiments/distribution_shift/main_experiment_metashift.py'
    base_script += ' --num-domains 2' 
    base_script += ' --details '+args.experiment_id
    
    data_indicator = ''
    data_indicator2 = ''
    if partial:
        domains_config = 'domains/cat_dog_pirm_1.json'
        data_indicator = 'p1'
        if args.experiment == 'SP':
            data_indicator2 = 'p2'
            domains_config2 = 'domains/cat_dog_pirm_2.json'
    else:
        domains_config = 'domains/cat_dog_irm.json'
        data_indicator = 'irm'
    train_experiment(args, base_script, domains_config, data_indicator)
    if args.experiment == 'SP' and partial:
        train_experiment(args, base_script, domains_config2, data_indicator2)

    # printing results
    print(f"label \t\t Test Acc. \tSize")
    read_results(args, data_indicator)
    if args.experiment == 'SP' and partial:
        read_results(args, data_indicator2)