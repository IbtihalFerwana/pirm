import sys
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='time alignment -- BERT IRM')
    parser.add_argument('--model', type=str, default='bert', help='language model for finetuning, bert or distilbert')
    parser.add_argument('--training_years', nargs='+', default = ['1980','1990','2000','2005','2010','2015'], help='a list of partitioned periods, indicated by the starting year')
    parser.add_argument('--testing_years', nargs='+', default = ['1980','1990','2000','2005','2010','2015'], help='a list of partitioned periods, indicated by the starting year')
    parser.add_argument('--output_file',  default = 'data/output_erm', help='output file to save results')
    parser.add_argument('--epochs',  type = int, default = 50, help='number of training epochs')
    parser.add_argument('--batch_size',  type = int, default = 2, help='batch size')
    parser.add_argument('--penalties',  nargs='+', default = [0.1, 1,10,100,1000,10000], help='irm penalties')

    args = parser.parse_args()

    training_years = args.training_years
    testing_years = args.testing_years
    output_file = args.output_file
    model_name = args.model
    epochs = args.epochs
    batch_size = args.batch_size
    penalties = args.penalties
    
    data_dir = 'sciERC_temporal'
    ib_lambda = 0
    method = 'irm'
    all_training_years = [training_years]

    base_script = 'python train_model_script.py'+' --batch_size '+str(batch_size)+' --ib_lambda '+str(ib_lambda)
    base_script += ' --method '+method
    base_script += ' --data_dir '+ data_dir
    base_script += ' --epochs '+str(epochs)
    base_script += ' --testing_years '+' '.join([str(ty) for ty in testing_years])
    base_script += ' --model '+model_name

    seeds = [0,1,2]
#     irm_penalties = [0.1, 1,10,100,1000,10000]
    irm_penalties = penalties
    # irm_penalties = [0.1]
    results_path = output_file+'results/'
    models_path = output_file+'saved_models/'
    for training_years in all_training_years:
        env_results_path = results_path+'train_'+'_'.join([str(ty) for ty in training_years])
        env_model_path = models_path+'train_'+'_'.join([str(ty) for ty in training_years])
        print("\tresults path: "+env_results_path)
        print("\tmodel path: "+env_model_path)
        print()
        if not os.path.exists(env_results_path):
            os.makedirs(env_results_path)
        if not os.path.exists(env_model_path):
            os.makedirs(env_model_path)
        for penalty in irm_penalties:
            for seed in seeds:
                print("\ttraining years: "+'_'.join([str(ty) for ty in training_years]))
                print("\ttesting years: "+'_'.join([str(ty) for ty in testing_years]))
                print("\tSeed: ", str(seed))
                print("\tirm penalty: ", str(penalty))
                script = base_script
                script+= ' --training_years '+' '.join([str(ty) for ty in training_years])+ ' --penalty_weight '+str(penalty)
                script+= ' --seed '+str(seed)
                script+= ' --output_file '+env_results_path+'/seed_'+str(seed)+'_penalty_'+str(penalty)
                script+= ' --model_path '+env_model_path+'/seed_'+str(seed)+'_penalty_'+str(penalty)
                print("\t"+script)
                print()
                os.system(script)
                print('\t***')
                print()
    