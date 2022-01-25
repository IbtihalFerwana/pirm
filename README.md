### updates
1. one can choose the batch size from arguments
2. This only works for samples size of 600 for envs [1980-1999, 2010-2016]

### Example
`!python train_model_script.py --batch_size 4 --training_years 1980 1990 --method "erm" --testing_years 2010 2015 --data_dir 'sciERC_temporal' --epochs 1 --output_file 'test_irm_model_2010_2016.csv'`

### TODO:
1. augmentation should happen in the data splitting

#### Time periods are defined as following (same for erm):
`input value` --> `included years`

`1980` --> `[1980 - 1989]`

`1990` --> `[1990 - 1999]`

`2000` --> `[2000 - 2004]`

`2005` --> `[2005 - 2009]`

`2010` --> `[2010 - 2014]`

`2015` --> `[2015 - 2016]`

#### The code works as following (same for erm): 
if you select `--testing_periods 1980 1990`, then it will test on each period (environment) sepearately and on both periods together (overall acc)
