### To run this branch for IRM code for BERT model:
1. get the dataset from: `IRM/code/time_embeddings/sciERC_temporal` folder in gdrive
2. use the running_examples.ipynb as a guidance for running the irm script
3. you can try different IRM penalty/regularizer values

#### Time periods are defined as following:
`input value` --> `included years`

`1980` --> `[1980 - 1989]`

`1990` --> `[1990 - 1999]`

`2000` --> `[2000 - 2004]`

`2005` --> `[2005 - 2009]`

`2010` --> `[2010 - 2014]`

`2015` --> `[2015 - 2016]`

#### The code works as following (same for erm): 
if you select `--testing_periods 1980 1990`, then it will test on each period (environment) sepearately and on both periods together (overall acc)
