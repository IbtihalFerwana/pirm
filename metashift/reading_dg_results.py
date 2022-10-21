###################################################
## This code was run on a jupyter notebook cell ##
## Not intended for now to run from terminal    ##
###################################################


# seeds = [i for i in range(100,1100,100)]
# details = f'{args.details}_seed_{str(args.seed)}_pen_{str(args.irm_lambda)}_lr_{args.lr}_anneal_{args.anneal}_penibirm_{args.ib_lambda}_ibanneal_{args.anneal_ib_irm}_bs_{args.batch_size}_partition_{split}.json'


exp_name = "cat_dog"
seeds = [0,1,2]
subsets = []
res = []
exps = ['irm','erm','ibirm']

anneals = [str(float(i)) for i in [40]]
pens = [str(float(i)) for i in [100]]

ib_pens = [str(float(i)) for i in [100]]
ib_anneals = [str(float(i)) for i in [40]]

## ibirm exp
for exp in ['ibirm','p1ibirm']:
    acc_dict = {}
    all_dict = {}
    all_dict["experiment"] = exp
    for pen in pens:
        for ib_pen in ib_pens:
            for anneal in anneals:
                for ib_anneal in ib_anneals:
                    for seed in seeds:
                        filename = f'cat_dog_results/{exp_name}_{exp}/{exp_name}_seed_{seed}_pen_{pen}_*_anneal_{anneal}_penibirm_{ib_pen}_ibanneal_{ib_anneal}*details_test.json'
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
                    d = {k:f'{np.mean(v):.3f}({np.std(v):.3f})' for k,v in acc_dict.items()}
                    all_dict.update(d)
                    res.append(all_dict)
                    print(acc_dict)
                
                               
## irm exp 
for exp in ['irm','p1']:
    acc_dict = {}
    all_dict = {}
    all_dict["experiment"] = exp
    for pen in pens:
        for anneal in anneals:
            for seed in seeds:
                filename = f'cat_dog_results/{exp_name}_{exp}/{exp_name}_seed_{seed}_pen_{pen}_*_anneal_{anneal}*details_test.json'
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
            d = {k:f'{np.mean(v):.3f}({np.std(v):.3f})' for k,v in acc_dict.items()}
            all_dict.update(d)
            res.append(all_dict)
            print(acc_dict)
for exp in ['erm','p1erm']:
    acc_dict = {}
    all_dict = {}
    all_dict["experiment"] = exp
    for seed in seeds:
        filename = f'cat_dog_results/{exp_name}_{exp}/{exp_name}_seed_{seed}_lr_*_bs_*details_test.json'
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
    d = {k:f'{np.mean(v):.3f}({np.std(v):.3f})' for k,v in acc_dict.items()}
    all_dict.update(d)
    res.append(all_dict)
    print(acc_dict)
print()