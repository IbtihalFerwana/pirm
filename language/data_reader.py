import argparse
import json
import os
import numpy as np


def get_years_list(min_year,max_year, period_size):
    if min_year < 1980:
        raise ValueError("min_year must be greater than or equal 1980")
    if max_year > 2016+1:
        raise ValueError("max_year must be less than or equal 2016")
    
    yrs_lst = list(np.arange(min_year, max_year, period_size))
    yrs_lst.append(max_year)
    
    yrs_string = [str(yr)+"-"+str(yrs_lst[i+1]-1) for i, yr in enumerate(yrs_lst[:-1])]
    
    return yrs_lst, yrs_string

def create_files_splits(yrs_lst, yrs_string, outdir):

    outfiles = {}
#     outdir = 'data'

    if not os.path.isdir(outdir):
        os.mkdir(outdir)


    for split in ['train','test', 'dev']:
        if not os.path.isdir(f'{outdir}/{split}'):
            os.mkdir(f'{outdir}/{split}')

        for i, yr in enumerate(yrs_lst[:-1]):
            outfiles[split + str(yr)] = open(f'{outdir}/{split}/{yrs_string[i]}.{split}.scores', 'w+')
    return outfiles

def split_data(raw_data, yrs_lst, outfiles):
    
    all_types_to_idx = {
    'Task': 0,
    'Method': 1,
    'Material': 2,
    'Metric': 3,
    'OtherScientificTerm': 4,
    'Generic': 5
    }
    
    
    all_data = {}
    for yr in yrs_lst[:-1]:
        all_data[yr] = []

    per_year_stats = {}

    for subdir, dirs, files in os.walk(f'{raw_data}'):
        print(subdir)

        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if file.endswith(".txt"):
    #             print(file)
                if "-" in file:
                    year = file.split("-")[0]
                    year = year[-2:]
                    if year[0] == '9' or year[0] == '8':
                        year = '19' + year
                    else:
                        year = '20' + year
                else:
                    year = file.split("_")[1]
                year = int(year)
    #             print(year)
                if year not in per_year_stats:
                    per_year_stats[year] = 0
                    # all_data[year] = []

                with open(filepath, 'r') as f:
                    text = f.read().replace("\n", " ").replace("\t", " ").strip()

                with open(filepath.replace('.txt', '.ann'), 'r') as f:
                    for line in f.readlines():
    #                     print("### line ",line)
                        line = line.replace('\n', '')
                        line_split = line.split('\t')
                        if len(line_split) == 3:
                            ent_type = line_split[1].split(" ")[0]
                            type_idx = all_types_to_idx[ent_type]
                            mention = line_split[2]
    #                         print(mention, ent_type)
                            row = {
                                'text': f"{mention} [SEP] {text}",
                                'labels': type_idx,
                                'year': year
                            }

                            per_year_stats[year] += 1
                            row = json.dumps(row)

                            for i, yr in enumerate(yrs_lst[:-1]):
                                if (year <= yrs_lst[i+1]) and (year >= yr):
                                    all_data[yr].append(row)
                                    continue

#         print(per_year_stats)
    print('-- Data Stats --')
    for year, rows in all_data.items():
        
        if len(rows) > 800:
            train_size = 600
        else:
            train_size = 400
        
        train = rows[:train_size]
        split_sizes = int((len(rows) - train_size)/2)
        test = rows[train_size:train_size + split_sizes]
        dev = rows[train_size + split_sizes:train_size + 2*split_sizes]
        
        print(f'\t year: {year} | train_size: {len(train)} | test_size: {len(test)} | dev_size: {len(dev)}')
        
        outfiles['train' + str(year)].write('\n'.join(train))
        outfiles['test' + str(year)].write('\n'.join(test))
        outfiles['dev' + str(year)].write('\n'.join(dev))
    
    print('files written')
    
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='time alignment -- read and split data')
    parser.add_argument('--raw_data', type=str, default='raw_data', help='raw data directory')
    parser.add_argument('--min_year', type=int, default=1980, help='minimum year in period range - included')                     
    parser.add_argument('--max_year', type=int, default=2017, help='maximum year in period range - excluded')
    parser.add_argument('--period_size', type=int, default=10, help='splitting years interval into equal sizes, left overs will have smaller size')
    parser.add_argument('--output_dir', type=str, default='sciERC_temporal', help='output directory') 
    args = parser.parse_args()
    min_year = args.min_year
    max_year = args.max_year
    period_size = args.period_size
    output_dir = args.output_dir
    raw_data = args.raw_data                                   
    
    
    yrs_lst, yrs_string = get_years_list(min_year, max_year, period_size)
    outfiles = create_files_splits(yrs_lst, yrs_string,output_dir)
    split_data(raw_data, yrs_lst, outfiles)