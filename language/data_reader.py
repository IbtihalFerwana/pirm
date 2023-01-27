import argparse
import json
import os
import numpy as np
import glob


def get_years_strings(yr_list):
    if yr_list[0] < 1980:
        raise ValueError("min_year must be greater than or equal 1980")
    if yr_list[-1] > 2016:
        raise ValueError("max_year must be less than or equal 2016")
    
    yrs_string = [str(yr)+"-"+str(yr_list[i+1]-1) for i, yr in enumerate(yr_list[:-1])]
    
    return yrs_string

def create_files_splits(yrs_lst, yrs_string, outdir):

    outfiles = {}
#     outdir = 'data'

    if not os.path.isdir(outdir):
        os.mkdir(outdir)


    for split in ['train','test', 'val']:
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
            filepath = subdir + os.sep + file

            if file.endswith(".txt"):
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
                if year not in per_year_stats:
                    per_year_stats[year] = 0

                with open(filepath, 'r') as f:
                    text = f.read().replace("\n", " ").replace("\t", " ").strip()

                with open(filepath.replace('.txt', '.ann'), 'r') as f:
                    for line in f.readlines():
                        line = line.replace('\n', '')
                        line_split = line.split('\t')
                        if len(line_split) == 3:
                            ent_type = line_split[1].split(" ")[0]
                            type_idx = all_types_to_idx[ent_type]
                            mention = line_split[2]
                            row = {
                                'text': f"{mention} [SEP] {text}",
                                'labels': type_idx,
                                'year': year
                            }

                            per_year_stats[year] += 1
                            row = json.dumps(row)

                            for i, yr in enumerate(yrs_lst[:-1]):
                                if (year < yrs_lst[i+1]) and (year >= yr):
                                    all_data[yr].append(row)
                                    continue

    #return per_year_stats
    print('-- Data Stats --')
    for year, rows in all_data.items():
        
        train_env_len = len(rows)
        a = np.arange(train_env_len)
        np.random.shuffle(a)

        train_split = 0.75
        test_split = 0.15

        train_indices = a[:int(train_env_len * train_split)]
        test_indices = a[int(train_env_len * train_split):int(train_env_len * (train_split+test_split))]
        val_indices = a[int(train_env_len * (train_split+test_split)):]


        print(train_env_len)

        train = [rows[i] for i in train_indices]
        test = [rows[i] for i in test_indices]
        val = [rows[i] for i in val_indices]
        
        print(f'\t year: {year} | train_size: {len(train)} | test_size: {len(test)} | dev_size: {len(val)}')
        
        outfiles['train' + str(year)].write('\n'.join(train))
        outfiles['test' + str(year)].write('\n'.join(test))
        outfiles['val' + str(year)].write('\n'.join(val))
    
    print('files written')
    return per_year_stats
    
def prep_scierc(raw_data):
    output_dir = f'{raw_data}/preprocessed_scierc'
    new_years_list = [1980, 1990, 2000, 2005, 2010, 2016]

    yrs_string = get_years_strings(new_years_list)
    print(yrs_string)

    outfiles = create_files_splits(new_years_list, yrs_string, output_dir)
    _ = split_data(raw_data, new_years_list, outfiles)
    return output_dir

def prep_aic(raw_data):
    output_dir = f'{raw_data}/preprocessed_aic'
    splits = ['train','test','dev']
    for split in splits:
        if split == 'train':
            g = glob.glob(f'{raw_data}/{split}/indivis/*')
        else:
            g = glob.glob(f'{raw_data}/{split}/*')
        if split == 'dev':
            new_split_dir = f'{output_dir}/val'
        else:
            new_split_dir = f'{output_dir}/{split}'
        
        if not os.path.exists(f'{new_split_dir}'):
            os.makedirs(f'{new_split_dir}')
        
        for g1 in g:
            print(g1)
            newfile_name = g1.split('/')[-1]
            newfile_name = f'{new_split_dir}/{newfile_name}'
            
            dict_list = []
            with open(g1,'r') as f:
                d = f.readlines()
                for txt in d:
                    doi_obj = eval(txt)
                    text = doi_obj['text']
                    label = doi_obj['label']
                    row = {"text":text,
                            "labels":label}
                    dict_list.append(json.dumps(row))
            with open(newfile_name, 'w') as fp:
                fp.write('\n'.join(dict_list))
            print("new file: ", newfile_name)
    return output_dir