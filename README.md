### Notes:
1. The `split_data.py` file creates data files with three splits of train, test and dev in a give output directory
2. The splits are equal sized periods for a given period range
3. The years range is given as [min, max), max year is exluded, e.g., one should use [1980, 2017) to include 2016.
4. The sample sizes at each time period are equalized with no augmentation, hence some would have different sample size. (This can be changed) 
5. One should use the raw data of SciERC dataset found in GDrive folder and here: http://nlp.cs.washington.edu/sciIE/

### To run the file, use the following arguments. 
```
--raw_data   raw data directory 
--min_year   minimum year in period range - included
--max_year   maximum year in period range - excluded
--period_size   for splitting years interval into equal sizes
--output_dir   output directory
```
