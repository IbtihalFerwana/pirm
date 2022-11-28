import json
import glob
import os
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='time alignment AIC -- read and change one key name')
    parser.add_argument('--raw_data', type=str, default='raw_data', help='raw data directory')
    parser.add_argument('--output_dir', type=str, default='data', help='output directory') 
    args = parser.parse_args()
    raw_dir = args.raw_data
    output_dir = args.output_dir

    splits = ['train','test','dev']
    # splits = ['train']
    for split in splits:
        if split == 'train':
            g = glob.glob(f'{raw_dir}/{split}/indivis/*')
        else:
            g = glob.glob(f'{raw_dir}/{split}/*')
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