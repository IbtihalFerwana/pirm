import sys
import os
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genome Dataset')
    parser.add_argument('--algorithm', type=str, default='IRM')
    parser.add_argument('--penalties_p1',  nargs='+', default = [100], help='irm penalties')
    parser.add_argument('--anneals_p1',  nargs='+', default = [100,200,500,1000], help='irm penalties')

    parser.add_argument('--penalties_p2',  nargs='+', default = [100], help='irm penalties')
    parser.add_argument('--anneals_p2',  nargs='+', default = [100,200,500,1000], help='irm penalties')

    parser.add_argument('--penalties_irm',  nargs='+', default = [100], help='irm penalties')
    parser.add_argument('--anneals_irm',  nargs='+', default = [100,200,500,1000], help='irm penalties')


    parser.add_argument('--data', type=str,help='data directory')
    parser.add_argument('--details', type=str,help='experiment name, must be unique in order to save and extract results')
    parser.add_argument('--pyfile', type=str,help='experiment py file e.g. main_experiment.py')
    parser.add_argument('--dataset_script', type=str)

    parser.add_argument('--exps', nargs='+', default=['p1','p2','irm'],help='experiment list')
    parser.add_argument('--data_exists', action='store_true',help='if data exists then use the provided data directory and do not regenerate')

    ### json files for partition groups
    irm_domains = 'irm_domains.json'
    pirm_domains = 'pirm_domains.json'

    args = parser.parse_args()
    # dataset_script = 'python '+args.dataset_script
    experiment_list = args.exps
    print("EXPERIMENT LIST: ", experiment_list)
    
    base_script = 'python '
    base_script += args.pyfile
    base_script += ' --num-domains 2' 

    base_script += ' --algorithm '+args.algorithm
    base_script += ' --details '+args.details
    
    

    shuffle_seeds = [42,43,44]
    # shuffle_seeds = [0,1,2]
    # shuffle_seeds = [2,3,4]
    seeds = [0,1,2]
    lrs = [5e-5]
    bss = [8]
    data_folder = args.data
    data_script = f'python {args.dataset_script}'
    if not args.data_exists:
        for seed in shuffle_seeds: 
            # data_script+=f' --data_folder data/MetaShift-subpopulation-shift-expD-iii-seed_{seed}'
            data_script+=f' --data_folder {data_folder}_seed_{seed}'
            data_script+=f' --shuffle_seed {seed}'
            os.system(data_script)
    
    if 'p1' in experiment_list:
        penalties = args.penalties_p1
        anneals = args.anneals_p1
        for penalty in penalties:
            print('irm_lambda: ', penalty)
            for lr in lrs:
                for bs in bss:
                    for anneal in anneals:
                        for sseed, seed in zip(shuffle_seeds,seeds):
                            np.random.seed(seed)
                            # os.system(dataset_script)
                            script = base_script
                            script += ' --domains '+pirm_domains
                            script += ' --experiment '+'pirm'
                            print('--seed', seed)
                            script += ' --seed '+str(seed)
                            script += ' --irm_lambda '+ str(penalty)
                            script += ' --lr '+str(lr)
                            script += ' --batch-size '+str(bs)
                            script += ' --anneal '+str(anneal)
                            
                            data_dir = 'results/'+args.details+'_p1'
                            os.makedirs(data_dir,exist_ok=True)
                            script_p1 = script+f' --data {data_folder}_seed_{sseed}/p1'
                            # script_p1 = script+f' --data {data_folder}/p1'
                            script_p1 += ' --results_file '+data_dir
                            os.system(script_p1)
                        
    # penalties = [500]
    # anneals = [100]
    if 'p2' in experiment_list:
        penalties = args.penalties_p2
        anneals = args.anneals_p2
        print(anneals)
        bss = [8]
        for penalty in penalties:
            print('irm_lambda: ', penalty)
            for lr in lrs:
                for bs in bss:
                    for anneal in anneals:
                        for sseed, seed in zip(shuffle_seeds,seeds):
                            np.random.seed(seed)
                            # os.system(dataset_script)
                            script = base_script
                            script += ' --domains '+pirm_domains
                            script += ' --experiment '+'pirm'
                            print('--seed', seed)
                            script += ' --seed '+str(seed)
                            script += ' --irm_lambda '+ str(penalty)
                            script += ' --lr '+str(lr)
                            script += ' --batch-size '+str(bs)
                            script += ' --anneal '+str(anneal)
                            
                            data_dir = 'results/'+args.details+'_p2'
                            os.makedirs(data_dir,exist_ok=True)
                            script_p2 = script+f' --data {data_folder}_seed_{sseed}/p2'
                            # script_p2 = script+f' --data {data_folder}/p2'
                            script_p2 += ' --results_file '+data_dir
                            os.system(script_p2)
    
    # exit(0)
    # penalties = [200]
    # anneals = [1000]
    if 'irm' in experiment_list:
        print("IRM only experiment")
        penalties = args.penalties_irm
        anneals = args.anneals_irm
        lrs = [5e-5]
        bss = [8]
        for penalty in penalties:
            print('irm_lambda: ', penalty)
            for lr in lrs:
                for bs in bss:
                    for anneal in anneals:
                        for sseed, seed in zip(shuffle_seeds,seeds):
                            np.random.seed(seed)
                            irm_script = base_script
                            irm_script += ' --domains '+irm_domains
                            irm_script += ' --experiment '+'irm'
                            irm_script += ' --seed '+str(seed)
                            irm_script += ' --irm_lambda '+ str(penalty)
                            irm_script += ' --lr '+str(lr)
                            irm_script += ' --batch-size '+str(bs)
                            irm_script += ' --anneal '+str(anneal)
                            data_dir = 'results/'+args.details+'_irm'
                            os.makedirs(data_dir,exist_ok=True)
                            # irm_script +=f' --data {data_folder}/irm'
                            irm_script +=f' --data {data_folder}_seed_{sseed}/irm'
                            irm_script += ' --results_file '+data_dir
                            os.system(irm_script)
                            
                            print('\t***')
                            print()
        
