The script can run both NLP tasks: `scierc` and `aic`

## SciERC
### Data
1. Download `raw_data` from [SciERC](http://nlp.cs.washington.edu/sciIE/).

##### Example: to run P-IRM (partitioned) on 3 envs

```
python pirm/language/train_test.py \
--batch_size 8 \
--penalty_weight 1000 \
--penalty_anneal_iters 40 \
--training_years 1990 2000 2005 \
--method "irm" \
--testing_years 2010 \
--train_conditioning 1990 2000 2005 \
--raw_data 'raw_data' \
--model 'bert' \
--epochs 1 \
--model_path 'partitioned_models_epochs80_bs8' \
--seed 0 \
--save_training_history True \
--save_best_model True \
--task 'scierc'
```

##### Example: to run P-IRM (conditioned) on 3 envs

```
python pirm/language/conditional_train_model_script_final_v1.py \
--batch_size 8 \
--penalty_weight 1000 \
--penalty_anneal_iters 40 \
--training_years 1980 1990 2000 2005 \
--method "irm" \
--testing_years 2010 \
--train_conditioning 1990 2000 2005 \
--raw_data 'raw_data' \
--model 'bert' \
--epochs 1 \
--model_path 'partitioned_models_epochs80_bs8' \
--seed 0 \
--save_training_history True \
--save_best_model True \
--task 'scierc'
```

## AIC
### Data
1. Download [aic_data](https://drive.google.com/drive/folders/1reoEybksfLi9np14klLLhXPB8wVJlpo0?usp=sharing) which is originally derived from [aic](https://github.com/Kel-Lu/time-waits-for-no-one/tree/main/data/aic)
2. Unzip `aic` directory

##### Example: to run P-IRM (conditioned) on 2 envs

```
python pirm/language/train_test.py \
--training_years 2006 2009 2012 2015 \
--method "irm" \
--testing_years 2018 \
--raw_data "aic_data" \
--model "bert" \
--save_training_history True \
--save_best_model True \
--penalty_anneal_iters 20 \
--penalty_weight 1000 \
--seed 100 \
--train_conditioning 2012 2015 \
--model_path 'aic_conditioned_ibirm_models_epochs40_bs8' \
--batch_size 8 \
--ib_lambda 0.1 \
--epochs 1 \
--task "aic" 
```
#### Time periods are defined as following (same for erm):
`input value` --> `included years`

`1980` --> `[1980 - 1989]`

`1990` --> `[1990 - 1999]`

`2000` --> `[2000 - 2004]`

`2005` --> `[2005 - 2009]`

`2010` --> `[2010 - 2014]`

`2015` --> `[2015 - 2016]`

## Specifications
|                  | SciERC Experiment | AIC Experiment |
|------------------|-------------------|----------------|
| Number of GPUs   | 4                 | 4              |
| Batch Size       | 8                 | 8              |
| Number of Epochs | 80                | 40             |
