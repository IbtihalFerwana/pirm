The script can run both NLP tasks: `scierc` and `aic`

## SciERC
### Data
We used data from [SciERC](http://nlp.cs.washington.edu/sciIE/). First we preprocessed it using `data_reader.py`. 

### Example 
```
python data_reader.py --raw_data 'raw_data' --output_dir 'sciERC_temporal/equal_split' --period_size 10
```


```
python conditional_train_model_script_final_v1.py 
--batch_size 8 \
--penalty_anneal_iters 30 \
--training_years 2000 \
--method "irm" \
--testing_years 2008 2013 \
--train_conditioning 2000 \
--data_dir 'sciERC_temporal' \
--data_split 'equal_split' \
--model 'bert' \
--epochs 1 \
--model_path 'conditional_equal_split_00_06' \
--seed 0 \
--save_training_history True \
--save_best_model True \
--task 'scierc'
```
## AIC
### Data
We used preprocessed data from [aic](https://github.com/Kel-Lu/time-waits-for-no-one/tree/main/data/aic), please follow the [lfs](https://git-lfs.github.com/) instructions to download AIC data from [aic](https://github.com/Kel-Lu/time-waits-for-no-one/tree/main/data/aic). We organized that data to match our scheme using `aic_data_reader.py`

### Example
```
python aic_data_reader.py --raw_data 'data' --output_dir 'data/preprocessed'
```

```
python conditional_train_model_script_final_v1.py \
--training_years 2006 2009 2012 2015 \
--method "irm" \
--testing_years 2018 \
--data_dir "data" \
--data_split "preprocessed" \
--model "bert" \
--save_training_history True \
--save_best_model True \
--penalty_anneal_iters 20 \
--penalty_weight 1000 \
--seed 100 \
--train_conditioning 2012 2015 \
--model_path 'aic_conditioned_ibirm_models_epochs40_bs8' \
--batch_size 8\
--ib_lambda 0.1 \
--epochs 40 \
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
