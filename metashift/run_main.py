import sys
import os
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genome Dataset')
    # parser.add_argument('--algorithm', type=str, default='IRM')

    parser.add_argument('--penalties_irm',  nargs='+', default = [100], help='irm penalties')
    parser.add_argument('--anneals_irm',  nargs='+', default = [100,200,500,1000], help='irm penalties')
    
    parser.add_argument('--details', type=str,help='experiment name, must be unique in order to save and extract results')
    parser.add_argument('--pyfile', type=str,help='experiment py file e.g. main_experiment.py')
    parser.add_argument('--dataset_script', type=str)

    parser.add_argument('--exps', nargs='+', default=['p1','p2','irm','p1erm','erm','p2erm','p1ibirm','p2ibirm','ibirm'],help='experiment list')
    parser.add_argument('--data_exists', action='store_true',help='if data exists then use the provided data directory and do not regenerate')
    parser.add_argument('--irm_data', default = '',type=str,help='if data exists then use the provided data directory and do not regenerate')
    parser.add_argument('--output_dir',default='train_ouptut')
    parser.add_argument('--data', type=str,help='data directory')
    parser.add_argument('--raw_results_folder', type=str,help='raw results directory')

    parser.add_argument('--penalties_ibirm',  nargs='+', default = [100], help='irm penalties')

    parser.add_argument('--anneals_ibirm',nargs='+', default = [100,200,500,1000], help='ib_irm anneals')

    ### json files for Sub-population Experiments
    irm_domains = 'irm_domains.json'
    pirm_1_domains = 'pirm_domains.json'
    pirm_2_domains = 'pirm_domains.json'

    ### json files for Domain Generalization Experiments
    # irm_domains = 'cat_dog_irm.json'
    # pirm_1_domains = 'cat_dog_pirm_1.json'
    # pirm_2_domains = 'cat_dog_pirm_2.json'

    # irm_domains = 'elephant_horse_irm.json'
    # pirm_1_domains = 'elephant_horse_pirm1.json'
    # pirm_2_domains = 'elephant_horse_pirm2.json'

    # irm_domains = 'bowl_cup_irm.json'
    # pirm_1_domains = 'bowl_cup_pirm_1.json'
    # pirm_2_domains = 'bowl_cup_pirm_2.json'

    # results_folder = 'raw_results_05_22'
    args = parser.parse_args()
    # dataset_script = 'python '+args.dataset_script
    results_folder = args.raw_results_folder
    experiment_list = args.exps
    
    # experiment_list = ['p1','p2','irm','p1erm','erm','p2erm','p1ibirm','p2ibirm','ibirm']
    # experiment_list = ['p1erm','p2erm']
    # experiment_list = ['p1','p2','irm','p1erm','p2erm','erm']
    # experiment_list = ['irm']
    print("EXPERIMENT LIST: ", experiment_list)
    base_script = 'python '
    base_script += args.pyfile
    base_script += ' --num-domains 2' 

    # base_script += ' --algorithm '+args.algorithm
    base_script += ' --details '+args.details
    base_script += ' --output_dir '+args.output_dir
    
    

    shuffle_seeds = [42]
    # seeds = [100,200,300,400,500,600,700,800,900,1000]
    seeds = [i for i in range(3)]
    # seeds = [42,43,44]
    # seeds = [100,200,300]
    lrs = [5e-5]
    bss = [8]
    data_folder = args.data

    print(data_folder)
    
    data_script = f'python {args.dataset_script}'
    if not args.data_exists:
        for seed in shuffle_seeds: 
            # data_script+=f' --data_folder data/MetaShift-subpopulation-shift-expD-iii-seed_{seed}'
            data_script+=f' --data_folder {data_folder}'
            data_script+=f' --shuffle_seed {seed}'
            os.system(data_script)
    
    if 'p1' in experiment_list:
        print("========= P1 experiment =========")
        penalties = args.penalties_irm
        anneals = args.anneals_irm
        for penalty in penalties:
            print('irm_lambda: ', penalty)
            for lr in lrs:
                for bs in bss:
                    for anneal in anneals:
                        for seed in seeds:
                            np.random.seed(seed)
                            # os.system(dataset_script)
                            script = base_script
                            script += ' --domains '+pirm_1_domains
                            script += ' --experiment '+'p1'
                            print('--seed', seed)
                            script += ' --seed '+str(seed)
                            script += ' --irm_lambda '+ str(penalty)
                            script += ' --lr '+str(lr)
                            script += ' --batch-size '+str(bs)
                            script += ' --anneal '+str(anneal)
                            script += ' --algorithm IRM'
                            
                            data_dir = f'{results_folder}/'+args.details+'_p1'
                            os.makedirs(data_dir,exist_ok=True)
                            script_p1 = script+f' --data {data_folder}/p1'
                            # script_p1 = script+f' --data {data_folder}/p1'
                            script_p1 += ' --results_file '+data_dir
                            os.system(script_p1)
                        
    # penalties = [500]
    # anneals = [100]
    if 'p2' in experiment_list:
        print("========= P2 experiment =========")
        penalties = args.penalties_irm
        anneals = args.anneals_irm
        print(anneals)
        bss = [8]
        for penalty in penalties:
            print('irm_lambda: ', penalty)
            for lr in lrs:
                for bs in bss:
                    for anneal in anneals:
                        for seed in seeds:
                            np.random.seed(seed)
                            # os.system(dataset_script)
                            script = base_script
                            script += ' --domains '+pirm_2_domains
                            script += ' --experiment '+'p2'
                            print('--seed', seed)
                            script += ' --seed '+str(seed)
                            script += ' --irm_lambda '+ str(penalty)
                            script += ' --lr '+str(lr)
                            script += ' --batch-size '+str(bs)
                            script += ' --anneal '+str(anneal)
                            script += ' --algorithm IRM'
                            
                            data_dir = f'{results_folder}/'+args.details+'_p2'
                            os.makedirs(data_dir,exist_ok=True)
                            script_p2 = script+f' --data {data_folder}/p2'
                            # script_p2 = script+f' --data {data_folder}/p2'
                            script_p2 += ' --results_file '+data_dir
                            os.system(script_p2)
    
    # exit(0)
    # penalties = [200]
    # anneals = [1000]
    # data_folder = 'data_no_shuffle/experiment_G_0_named_tests_seed_42'
    # if args.irm_data != '':
    #     data_folder = args.irm_data
    # else:
    #     data_folder = f'{data_folder}_seed_*/'
    if 'irm' in experiment_list:
        print("========= IRM experiment =========")
        penalties = args.penalties_irm
        anneals = args.anneals_irm
        lrs = [5e-5]
        bss = [8]
        for penalty in penalties:
            print('irm_lambda: ', penalty)
            for lr in lrs:
                for bs in bss:
                    for anneal in anneals:
                        for seed in seeds:
                            np.random.seed(seed)
                            irm_script = base_script
                            irm_script += ' --domains '+irm_domains
                            irm_script += ' --experiment '+'irm'
                            irm_script += ' --seed '+str(seed)
                            irm_script += ' --irm_lambda '+ str(penalty)
                            irm_script += ' --lr '+str(lr)
                            irm_script += ' --batch-size '+str(bs)
                            irm_script += ' --anneal '+str(anneal)
                            irm_script += ' --algorithm IRM'
                            data_dir = f'{results_folder}/'+args.details+'_irm'
                            os.makedirs(data_dir,exist_ok=True)
                            irm_script +=f' --data {data_folder}/irm'
                            # irm_script +=f' --data {data_folder}irm'
                            irm_script += ' --results_file '+data_dir
                            os.system(irm_script)
                            
                            print('\t***')
                            print()
    if 'erm' in experiment_list:
        print("========= ERM experiment =========")
        lrs = [5e-5]
        bss = [8]
        for lr in lrs:
            for bs in bss:
                for seed in seeds:
                    np.random.seed(seed)
                    irm_script = base_script
                    irm_script += ' --domains '+irm_domains
                    irm_script += ' --experiment '+'erm'
                    irm_script += ' --seed '+str(seed)
                    irm_script += ' --lr '+str(lr)
                    irm_script += ' --batch-size '+str(bs)
                    irm_script += ' --algorithm ERM'
                    data_dir = f'{results_folder}/'+args.details+'_erm'
                    os.makedirs(data_dir,exist_ok=True)
                    irm_script +=f' --data {data_folder}/irm'
                    # irm_script +=f' --data {data_folder}irm'
                    irm_script += ' --results_file '+data_dir
                    os.system(irm_script)
                    
                    print('\t***')
                    print()
    if 'p1erm' in experiment_list:
        print("========= P1 ERM experiment =========")
    
        for lr in lrs:
            for bs in bss:
                for seed in seeds:
                    np.random.seed(seed)
                    # os.system(dataset_script)
                    script = base_script
                    script += ' --domains '+pirm_1_domains
                    script += ' --experiment '+'p1erm'
                    print('--seed', seed)
                    script += ' --seed '+str(seed)
                    script += ' --lr '+str(lr)
                    script += ' --batch-size '+str(bs)
                    script += ' --algorithm ERM'
                    
                    data_dir = f'{results_folder}/'+args.details+'_p1erm'
                    os.makedirs(data_dir,exist_ok=True)
                    script_p1 = script+f' --data {data_folder}/p1'
                    # script_p1 = script+f' --data {data_folder}/p1'
                    script_p1 += ' --results_file '+data_dir
                    os.system(script_p1)
                        
    if 'p2erm' in experiment_list:
        print("========= P2 ERM experiment =========")
        bss = [8]
        for lr in lrs:
            for bs in bss:
                for seed in seeds:
                    np.random.seed(seed)
                    # os.system(dataset_script)
                    script = base_script
                    script += ' --domains '+pirm_2_domains
                    script += ' --experiment '+'p2erm'
                    print('--seed', seed)
                    script += ' --seed '+str(seed)
                    script += ' --lr '+str(lr)
                    script += ' --batch-size '+str(bs)
                    script += ' --algorithm ERM'
                    
                    data_dir = f'{results_folder}/'+args.details+'_p2erm'
                    os.makedirs(data_dir,exist_ok=True)
                    script_p2 = script+f' --data {data_folder}/p2'
                    # script_p2 = script+f' --data {data_folder}/p2'
                    script_p2 += ' --results_file '+data_dir
                    os.system(script_p2)

    if 'p1ibirm' in experiment_list:
        print("========= P1 IB_IRM experiment =========")
        penalties = args.penalties_irm
        anneals = args.anneals_irm
        anneals_ibirm = args.anneals_ibirm
        penalties_ibirm = args.penalties_ibirm

        for penalty in penalties:
            for ib_penalty in penalties_ibirm:
                for lr in lrs:
                    for bs in bss:
                        for anneal in anneals:
                            for anneal_ib in anneals_ibirm:
                                for seed in seeds:
                                    np.random.seed(seed)
                                    # os.system(dataset_script)
                                    script = base_script
                                    script += ' --domains '+pirm_1_domains
                                    script += ' --experiment '+'p1ibirm'
                                    print('--seed', seed)
                                    script += ' --seed '+str(seed)
                                    script += ' --irm_lambda '+ str(penalty)
                                    script += ' --lr '+str(lr)
                                    script += ' --batch-size '+str(bs)
                                    script += ' --anneal '+str(anneal)
                                    script += ' --algorithm IB_IRM'
                                    script += ' --ib_penalty_anneal_iters '+str(anneal_ib)
                                    script += ' --ib_lambda '+str(ib_penalty)
                                    
                                    data_dir = f'{results_folder}/'+args.details+'_p1ibirm'
                                    os.makedirs(data_dir,exist_ok=True)
                                    script_p1 = script+f' --data {data_folder}/p1'
                                    # script_p1 = script+f' --data {data_folder}/p1'
                                    script_p1 += ' --results_file '+data_dir
                                    os.system(script_p1)
                            
    if 'p2ibirm' in experiment_list:
        print("========= P2 IB_IRM experiment ========= ")
        penalties = args.penalties_irm
        anneals = args.anneals_irm
        anneals_ibirm = args.anneals_ibirm
        penalties_ibirm = args.penalties_ibirm
        bss = [8]
        for penalty in penalties:
            for ib_penalty in penalties_ibirm:
                for lr in lrs:
                    for bs in bss:
                        for anneal in anneals:
                            for anneal_ib in anneals_ibirm:
                                for seed in seeds:
                                    np.random.seed(seed)
                                    # os.system(dataset_script)
                                    script = base_script
                                    script += ' --domains '+pirm_2_domains
                                    script += ' --experiment '+'p2ibirm'
                                    print('--seed', seed)
                                    script += ' --seed '+str(seed)
                                    script += ' --irm_lambda '+ str(penalty)
                                    script += ' --lr '+str(lr)
                                    script += ' --batch-size '+str(bs)
                                    script += ' --anneal '+str(anneal)
                                    script += ' --algorithm IB_IRM'
                                    script += ' --ib_penalty_anneal_iters '+str(anneal_ib)
                                    script += ' --ib_lambda '+str(ib_penalty)
                                    
                                    data_dir = f'{results_folder}/'+args.details+'_p2ibirm'
                                    os.makedirs(data_dir,exist_ok=True)
                                    script_p2 = script+f' --data {data_folder}/p2'
                                    # script_p2 = script+f' --data {data_folder}/p2'
                                    script_p2 += ' --results_file '+data_dir
                                    os.system(script_p2)
        
    if 'ibirm' in experiment_list:
        print("========= IB_IRM experiment =========")
        penalties = args.penalties_irm
        anneals = args.anneals_irm
        anneals_ibirm = args.anneals_ibirm
        penalties_ibirm = args.penalties_ibirm

        lrs = [5e-5]
        bss = [8]
        for penalty in penalties:
            for ib_penalty in penalties_ibirm:
                print('irm_lambda: ', penalty)
                for lr in lrs:
                    for bs in bss:
                        for anneal in anneals:
                            for anneal_ib in anneals_ibirm:
                                for seed in seeds:
                                    np.random.seed(seed)
                                    irm_script = base_script
                                    irm_script += ' --domains '+irm_domains
                                    irm_script += ' --experiment '+'ibirm'
                                    irm_script += ' --seed '+str(seed)
                                    irm_script += ' --irm_lambda '+ str(penalty)
                                    irm_script += ' --lr '+str(lr)
                                    irm_script += ' --batch-size '+str(bs)
                                    irm_script += ' --anneal '+str(anneal)
                                    irm_script += ' --algorithm IB_IRM'
                                    irm_script += ' --ib_penalty_anneal_iters '+str(anneal_ib)
                                    irm_script += ' --ib_lambda '+str(ib_penalty)

                                    data_dir = f'{results_folder}/'+args.details+'_ibirm'
                                    os.makedirs(data_dir,exist_ok=True)
                                    irm_script +=f' --data {data_folder}/irm'
                                    # irm_script +=f' --data {data_folder}irm'
                                    irm_script += ' --results_file '+data_dir
                                    os.system(irm_script)
                                    
                                    print('\t***')
                                    print()
                
    # if 'irm_fix' in experiment_list:
    #     print("Fixing IRM experiment")
    #     penalties = args.penalties_irm
    #     anneals = args.anneals_irm
    #     lrs = [5e-5]
    #     bss = [8]
    #     for penalty in penalties:
    #         print('irm_lambda: ', penalty)
    #         for lr in lrs:
    #             for bs in bss:
    #                 for anneal in anneals:
    #                     for seed in seeds:
    #                         np.random.seed(seed)
    #                         irm_script = base_script
    #                         irm_script += ' --domains '+irm_domains
    #                         irm_script += ' --experiment '+'irm'
    #                         irm_script += ' --seed '+str(seed)
    #                         irm_script += ' --irm_lambda '+ str(penalty)
    #                         irm_script += ' --lr '+str(lr)
    #                         irm_script += ' --batch-size '+str(bs)
    #                         irm_script += ' --anneal '+str(anneal)
    #                         irm_script += ' --algorithm '+args.algorithm
    #                         data_dir = f'{results_folder}/'+args.details+'_irm'
    #                         os.makedirs(data_dir,exist_ok=True)
    #                         irm_script +=f' --data {data_folder}'
    #                         # irm_script +=f' --data {data_folder}irm'
    #                         irm_script += ' --results_file '+data_dir
    #                         os.system(irm_script)
                            
    #                         print('\t***')
    #                         print()
    #     print("ERM experiment")
    #     penalties = args.penalties_irm
    #     anneals = args.anneals_irm
    #     lrs = [5e-5]
    #     bss = [8]
    #     for lr in lrs:
    #         for bs in bss:
    #             for seed in seeds:
    #                 np.random.seed(seed)
    #                 irm_script = base_script
    #                 irm_script += ' --domains '+irm_domains
    #                 irm_script += ' --experiment '+'irm'
    #                 irm_script += ' --seed '+str(seed)
    #                 irm_script += ' --lr '+str(lr)
    #                 irm_script += ' --batch-size '+str(bs)
    #                 irm_script += ' --algorithm ERM'
    #                 data_dir = f'{results_folder}/'+args.details+'_erm'
    #                 os.makedirs(data_dir,exist_ok=True)
    #                 irm_script +=f' --data {data_folder}'
    #                 # irm_script +=f' --data {data_folder}irm'
    #                 irm_script += ' --results_file '+data_dir
    #                 os.system(irm_script)
                    
    #                 print('\t***')
    #                 print()
    # if 'ibirm_fix' in experiment_list:
    #     print("Fixing IB_IRM experiment")
    #     penalties = args.penalties_irm
    #     anneals = args.anneals_irm
    #     lrs = [5e-5]
    #     bss = [8]
    #     for penalty in penalties:
    #         print('irm_lambda: ', penalty)
    #         for lr in lrs:
    #             for bs in bss:
    #                 for anneal in anneals:
    #                     for seed in seeds:
    #                         np.random.seed(seed)
    #                         irm_script = base_script
    #                         irm_script += ' --domains '+irm_domains
    #                         irm_script += ' --experiment '+'irm'
    #                         irm_script += ' --seed '+str(seed)
    #                         irm_script += ' --irm_lambda '+ str(penalty)
    #                         irm_script += ' --lr '+str(lr)
    #                         irm_script += ' --batch-size '+str(bs)
    #                         irm_script += ' --anneal '+str(anneal)
    #                         irm_script += ' --algorithm IB_IRM'
    #                         irm_script += ' --ib_penalty_anneal_iters '+str(anneal_ib_irm)
    #                         data_dir = f'{results_folder}/'+args.details+'_irm'
    #                         os.makedirs(data_dir,exist_ok=True)
    #                         irm_script +=f' --data {data_folder}'
    #                         # irm_script +=f' --data {data_folder}irm'
    #                         irm_script += ' --results_file '+data_dir
    #                         os.system(irm_script)
                            
    #                         print('\t***')
    #                         print()
                            
    #     # print("DRO experiment")
    #     # penalties = args.penalties_irm
    #     # anneals = args.anneals_irm
    #     # lrs = [5e-5]
    #     # bss = [8]
    #     # for penalty in penalties:
    #     #     print('irm_lambda: ', penalty)
    #     #     for lr in lrs:
    #     #         for bs in bss:
    #     #             for anneal in anneals:
    #     #                 for seed in seeds:
    #     #                     np.random.seed(seed)
    #     #                     irm_script = base_script
    #     #                     irm_script += ' --domains '+irm_domains
    #     #                     irm_script += ' --experiment '+'irm'
    #     #                     irm_script += ' --seed '+str(seed)
    #     #                     irm_script += ' --irm_lambda '+ str(penalty)
    #     #                     irm_script += ' --lr '+str(lr)
    #     #                     irm_script += ' --batch-size '+str(bs)
    #     #                     irm_script += ' --anneal '+str(anneal)
    #     #                     irm_script += ' --algorithm GroupDRO'
    #     #                     data_dir = f'{results_folder}/'+args.details+'_groupdro'
    #     #                     os.makedirs(data_dir,exist_ok=True)
    #     #                     irm_script +=f' --data {data_folder}/irm'
    #     #                     # irm_script +=f' --data {data_folder}irm'
    #     #                     irm_script += ' --results_file '+data_dir
    #     #                     os.system(irm_script)
                            
    #     #                     print('\t***')
    #     #                     print()
        
    #     # print("CORAL experiment")
    #     # penalties = args.penalties_irm
    #     # anneals = args.anneals_irm
    #     # lrs = [5e-5]
    #     # bss = [8]
    #     # for penalty in penalties:
    #     #     print('irm_lambda: ', penalty)
    #     #     for lr in lrs:
    #     #         for bs in bss:
    #     #             for anneal in anneals:
    #     #                 for seed in seeds:
    #     #                     np.random.seed(seed)
    #     #                     irm_script = base_script
    #     #                     irm_script += ' --domains '+irm_domains
    #     #                     irm_script += ' --experiment '+'irm'
    #     #                     irm_script += ' --seed '+str(seed)
    #     #                     irm_script += ' --irm_lambda '+ str(penalty)
    #     #                     irm_script += ' --lr '+str(lr)
    #     #                     irm_script += ' --batch-size '+str(bs)
    #     #                     irm_script += ' --anneal '+str(anneal)
    #     #                     irm_script += ' --algorithm CORAL'
    #     #                     data_dir = f'{results_folder}/'+args.details+'_coral'
    #     #                     os.makedirs(data_dir,exist_ok=True)
    #     #                     irm_script +=f' --data {data_folder}/irm'
    #     #                     # irm_script +=f' --data {data_folder}irm'
    #     #                     irm_script += ' --results_file '+data_dir
    #     #                     os.system(irm_script)
                            
    #     #                     print('\t***')
    #                         # print()
    
    #     # print("CDANN experiment")
    #     # penalties = args.penalties_irm
    #     # anneals = args.anneals_irm
    #     # lrs = [5e-5]
    #     # bss = [8]
    #     # for penalty in penalties:
    #     #     print('irm_lambda: ', penalty)
    #     #     for lr in lrs:
    #     #         for bs in bss:
    #     #             for anneal in anneals:
    #     #                 for seed in seeds:
    #     #                     np.random.seed(seed)
    #     #                     irm_script = base_script
    #     #                     irm_script += ' --domains '+irm_domains
    #     #                     irm_script += ' --experiment '+'irm'
    #     #                     irm_script += ' --seed '+str(seed)
    #     #                     irm_script += ' --irm_lambda '+ str(penalty)
    #     #                     irm_script += ' --lr '+str(lr)
    #     #                     irm_script += ' --batch-size '+str(bs)
    #     #                     irm_script += ' --anneal '+str(anneal)
    #     #                     irm_script += ' --algorithm CDANN'
    #     #                     data_dir = f'{results_folder}/'+args.details+'_cdann'
    #     #                     os.makedirs(data_dir,exist_ok=True)
    #     #                     irm_script +=f' --data {data_folder}/irm'
    #     #                     # irm_script +=f' --data {data_folder}irm'
    #     #                     irm_script += ' --results_file '+data_dir
    #     #                     os.system(irm_script)
                            
    #     #                     print('\t***')
    #     #                     print()
        
