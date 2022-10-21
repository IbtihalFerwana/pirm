###################################################
## This code was run on a jupyter notebook cell ##
## Not intended for now to run from terminal    ##
###################################################



# seeds = [i for i in range(100,1100,100)]
# details = f'{args.details}_seed_{str(args.seed)}_pen_{str(args.irm_lambda)}_lr_{args.lr}_anneal_{args.anneal}_penibirm_{args.ib_lambda}_ibanneal_{args.anneal_ib_irm}_bs_{args.batch_size}_partition_{split}.json'

exp_name = "experiment_exp2_B"
results_folder = "raw_results_sps_cats_dogs_09_18"
seeds = [0,1,2]
subsets = []
res = []
exps = ['irm','p1','erm','p1erm','ibirm','p1ibirm']

anneals = [str(float(i)) for i in [20,40]]
pens = [str(float(i)) for i in [10,100,1000]]

ib_pens = [str(float(i)) for i in [10,100,1000]]
ib_anneals = [str(float(i)) for i in [20,40]]


## ibirm exp
for exp in ['ibirm','p1ibirm','p2ibirm']:
    print("========"+exp+"=======")
    for pen in pens:
        for ib_pen in ib_pens:
            for anneal in anneals:
                for ib_anneal in ib_anneals:
                    acc_dict = {}
                    all_dict = {}
                    acc_dict['val_acc'] = []
                    all_dict["experiment"] = exp
                    for seed in seeds:
                        filename = f'{results_folder}/{exp_name}_{exp}/{exp_name}_seed_{seed}_pen_{pen}_*_anneal_{anneal}_penibirm_{ib_pen}_ibanneal_{ib_anneal}*details_test.json'
                        for g in glob.glob(filename):
                            with open(g,'r') as f:
                                df_all = json.load(f)
                            for df in df_all:
                                acc = float(df['acc'])
                                gr = df['groups_local']
                                if gr in acc_dict:
                                    acc_dict[gr].append(acc)
                                elif gr not in acc_dict:
                                    acc_dict[gr] = [acc]
                        # Validation
                        filename = f'{results_folder}/{exp_name}_{exp}/{exp_name}_seed_{seed}_pen_{pen}_*_anneal_{anneal}_penibirm_{ib_pen}_ibanneal_{ib_anneal}*details_val.json'
                        for g in glob.glob(filename):
                            with open(g,'r') as f:
                                df_all = json.load(f)
                            val_acc = 0
                            for vali in df_all:
                                val_acc+=float(vali['acc'])
                            acc_dict['val_acc'].append(val_acc/4)
                    d = {k:f'{np.mean(v):.3f}({np.std(v):.3f})' for k,v in acc_dict.items()}
#                     d = {k:f'{np.mean(v):.3f}' for k,v in acc_dict.items()}
                    all_dict['irm_pen'] = pen
                    all_dict['irm_anneal'] = anneal
                    all_dict['ibirm_pen'] = ib_pen
                    all_dict['ibim_anneal'] = ib_anneal
                    all_dict.update(d)
                    res.append(all_dict)
#                     print(acc_dict)
                    print("\t\tacc dict size: ", len(acc_dict['val_acc']))
    
                    if len(acc_dict['val_acc']) <=1 :
                        print("irm_pen: ", pen)
                        print("irm_anneal: ",anneal)
                        print("ibirm_pen: ",ib_pen)
                        print("ibim_anneal: ",ib_anneal)
                
                               
# irm exp 
for exp in ['irm','p1','p2']:
    print("========"+exp+"=======")
    for pen in pens:
        for anneal in anneals:
            acc_dict = {}
            acc_dict['val_acc'] = []
            all_dict = {}
            all_dict["experiment"] = exp
            for seed in seeds:
                filename = f'{results_folder}/{exp_name}_{exp}/{exp_name}_seed_{seed}_pen_{pen}_*_anneal_{anneal}*details_test.json'
                for g in glob.glob(filename):
                    with open(g,'r') as f:
                        df_all = json.load(f)
                    for df in df_all:
                        acc = float(df['acc'])

                        gr = df['groups_local']
                        if gr in acc_dict:
                            acc_dict[gr].append(acc)
                        elif gr not in acc_dict:
                            acc_dict[gr] = [acc]
                filename = f'{results_folder}/{exp_name}_{exp}/{exp_name}_seed_{seed}_pen_{pen}_*_anneal_{anneal}*details_val.json'
                for g in glob.glob(filename):
                    with open(g,'r') as f:
                        df_all = json.load(f)
                    val_acc = 0
                    for vali in df_all:
                        val_acc+=float(vali['acc'])
                    acc_dict['val_acc'].append(val_acc/4)
            d = {k:f'{np.mean(v):.3f}({np.std(v):.3f})' for k,v in acc_dict.items()}
#             d = {k:f'{np.mean(v):.3f}' for k,v in acc_dict.items()}
            all_dict['irm_pen'] = pen
            all_dict['irm_anneal'] = anneal
            all_dict['ibirm_pen'] = 0
            all_dict['ibim_anneal'] = 0
            all_dict.update(d)
            res.append(all_dict)
            print("\t\tacc dict size: ", len(acc_dict['val_acc']))
for exp in ['erm','p1erm','p2erm']:
    print("========"+exp+"=======")
    acc_dict = {}
    acc_dict['val_acc'] = []
    all_dict = {}
    all_dict["experiment"] = exp
    for seed in seeds:
        filename = f'{results_folder}/{exp_name}_{exp}/{exp_name}_seed_{seed}_lr_*_bs_*details_test.json'
        for g in glob.glob(filename):
            with open(g,'r') as f:
                df_all = json.load(f)
            for df in df_all:
                acc = float(df['acc'])

                gr = df['groups_local']
                if gr in acc_dict:
                    acc_dict[gr].append(acc)
                elif gr not in acc_dict:
                    acc_dict[gr] = [acc]
        filename = f'{results_folder}/{exp_name}_{exp}/{exp_name}_seed_{seed}_lr_*_bs_*details_val.json'
        for g in glob.glob(filename):
            with open(g,'r') as f:
                df_all = json.load(f)
            val_acc = 0
            for vali in df_all:
                val_acc+=float(vali['acc'])
            acc_dict['val_acc'].append(val_acc/4)
    d = {k:f'{np.mean(v):.3f}({np.std(v):.3f})' for k,v in acc_dict.items()}
#     d = {k:f'{np.mean(v):.3f}' for k,v in acc_dict.items()}
    all_dict['irm_pen'] = 0
    all_dict['irm_anneal'] = 0
    all_dict['ibirm_pen'] = 0
    all_dict['ibim_anneal'] = 0
    all_dict.update(d)
    res.append(all_dict)
    print("\t\tacc dict size: ", len(acc_dict['val_acc']))
print()
df = pd.DataFrame(res)
new_df = pd.DataFrame()
for exp in ['ibirm','p1ibirm','p2ibirm','irm','p1','p2','erm','p1erm','p2erm']:
    df2 = df[df['experiment']==exp]
    min_val = df2['val_acc'].max()
    if min_val != 'nan':
        r = df2[df2['val_acc'] ==min_val]
        new_df = pd.concat([new_df,r])
new_df.to_csv(f"{results_folder}/{exp_name}_results_all_algs.csv")