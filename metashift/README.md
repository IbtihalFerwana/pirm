## Data
We used data from the [MetaShift](https://github.com/Weixin-Liang/MetaShift) project, follow the same procedure to download the [Visual Genome](https://github.com/Weixin-Liang/MetaShift#download-visual-genome) data. 
## Domain Generalization Experiment
1. Create the dataset using `dataset\domain_generalization_cat_dog_pirm_ii.py`
 
#### Example: 
- creating dataset with no overlap between environments `!python dataset/domain_generalization_cat_dog_pirm_ii.py --dataset_name '/data/Domain-Generalization-Cat-Dog-pirmii-exp1-A' --add_p 0`

- The output directory looks like following
```
/data/Domain-Generalization-Cat-Dog-pirmii-exp1-A

├── p1
    ├── imageID_to_group.pkl
    ├── train/
        ├── cat/
        ├── dog/ 
    ├── test/
        ├── cat/
        ├── dog/ 
    ├── val_out_of_domain/
        ├── cat/
        ├── dog/ 
├── p2
    ├── imageID_to_group.pkl
    ├── train/
    ├── test/
    ├── val_out_of_domain/
├── irm
    ├── imageID_to_group.pkl
    ├── train/
    ├── test/
    ├── val_out_of_domain/
 ```
 
 2. Run experiments with grid search on `irm` and `ibirm` penalties and annealing iteration values
 ```
 python run_main_experiment.py \
    --pyfile 'experiments/distribution_shift/main_experiment_metashift.py' \
    --penalties_irm 10 100 1000 \
    --penalties_ibirm 10 100 1000 \
    --anneals_irm 20 40 \
    --anneals_ibirm 20 40 \
    --raw_results_folder 'raw_results_dg_cats_dogs' \
    --data 'data/MetaShift/Domain-Generalization-Cat-Dog-pirmii-exp1-A' \
    --output_dir train_outputs/experiment_exp1-A \
    --exps irm p1 ibirm p1ibirm erm p1erm
    
 ```
