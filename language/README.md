### updates
1. The script can run both NLP tasks: `scierc` and `aic`

### Example
```
!python pirm/conditional_train_model_script_final_v1.py 
--batch_size 2 \
--penalty_anneal_iters 30 \
--training_years 2000 \
--method "irm" \
--testing_years 2008 2013 \
--train_conditioning 2000 \
--data_dir 'sciERC_temporal' \
--model 'bert' \
--epochs 1 \
--data_split 'equal_split' \
--model_path 'conditional_equal_split_00_06' \
--seed 0 \
--save_training_history True \
--save_best_model True \
--task 'scierc'
```

#### Time periods are defined as following (same for erm):
`input value` --> `included years`

`1980` --> `[1980 - 1989]`

`1990` --> `[1990 - 1999]`

`2000` --> `[2000 - 2004]`

`2005` --> `[2005 - 2009]`

`2010` --> `[2010 - 2014]`

`2015` --> `[2015 - 2016]`


### TODO:
1. add the option of gpt-2
