"""
Generate MetaDataset with train/test split 
"""

# CUSTOM_SPLIT_DATASET_FOLDER = 'data/Domain-Generalization-Cat-Dog-pirmii-exp4-C'

import pandas as pd 
import seaborn as sns

import pickle
import numpy as np
import json, re, math
from collections import Counter, defaultdict
from itertools import repeat
import pprint
import os, errno
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil # for copy files
import networkx as nx # graph vis
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import networkx.algorithms.community as nx_comm
import itertools
import random
import argparse
import Constants
import networkx as nx
from sklearn.model_selection import train_test_split
IMAGE_DATA_FOLDER          = Constants.IMAGE_DATA_FOLDER

from generate_full_MetaShift import preprocess_groups, copy_image_for_subject
from generate_full_MetaShift_exp1 import build_subset_graph


def print_communities(subject_data, node_name_to_img_id, trainsg_dupes, subject_str):
    ##################################
    # Community detection 
    ##################################
    G, pars_sc, Adjacency_matrix = build_subset_graph(subject_data, node_name_to_img_id, trainsg_dupes, subject_str,seed=0)
    
    import networkx.algorithms.community as nxcom

    # Find the communities
    communities = sorted(nxcom.greedy_modularity_communities(G), key=len, reverse=True)
    # Count the communities
    print(f"The graph has {len(communities)} communities.")
    for community in communities:
        community_merged = set()
        for node_str in community:
            node_str = node_str.replace('\n', '')
            node_image_IDs = node_name_to_img_id[node_str]
            community_merged.update(node_image_IDs)
            # print(node_str , len(node_image_IDs), end=';')

        print('total size:',len(community_merged))
        community_set = set([ x.replace('\n', '') for x in community])
        print(community_set, '\n\n')
    return G 



def parse_dataset_scheme(dataset_scheme, node_name_to_img_id,dataset_folder=None, exclude_img_id=set(), split='test', copy=True, trunc_size=2500):
    """
    exclude_img_id contains both trainsg_dupes and test images that we do not want to leak 
    """
    community_name_to_img_id = defaultdict(set)
    print(" START *** exlude image length: ", len(exclude_img_id))
    all_img_id = set()
    exclude_img_id_local = set()
    exclude_img_id_local = exclude_img_id.copy()
    # print("***** split ****", split)
    ##################################
    # Iterate subject_str: e.g., cat
    ##################################
    for subject_str in dataset_scheme:        
        ##################################
        # Iterate community_name: e.g., cat(sofa)
        ##################################
        for community_name in dataset_scheme[subject_str]:
            ##################################
            # Iterate node_name: e.g., 'cat(cup)', 'cat(sofa)', 'cat(chair)'
            ##################################
            for node_name in dataset_scheme[subject_str][community_name]:
                # print("node name: ", node_name)
                # print("exlude image length: ", len(exclude_img_id_local))
                community_name_to_img_id[community_name].update(node_name_to_img_id[node_name] - exclude_img_id_local)
                # if split == 'train':
                # exclude_img_id_local.update(all_img_id)
                # print("All images length: ", len(all_img_id))
            community_name_to_img_id[community_name] = set(shuffle_and_truncate(community_name_to_img_id[community_name], trunc_size))
            # community_name_to_img_id[community_name] = set(sorted(community_name_to_img_id[community_name])[:trunc_size])
            # community_name_to_img_id[community_name] = set(sorted(community_name_to_img_id[community_name]))
            exclude_img_id_local.update(community_name_to_img_id[community_name])
            all_img_id.update(community_name_to_img_id[community_name])
            # if copy:
            # print(community_name, 'Size:', len(community_name_to_img_id[community_name]) )
            


        ##################################
        # Iterate community_name: e.g., cat(sofa)
        ##################################
        if copy:
            root_folder = os.path.join(dataset_folder, split)
            copy_image_for_subject(root_folder, subject_str, dataset_scheme[subject_str], community_name_to_img_id, trainsg_dupes=set(), use_symlink=False) # use False to share 

    return community_name_to_img_id, all_img_id


def parse_dataset_scheme_preexisting_data(dataset_scheme, node_name_to_img_id,dataset_folder=None, exclude_img_id=set(),include_img_id=set(), additional_comms = set(), max_additional_size = 0,split='test', copy=True, trunc_size=2500):
    """
    exclude_img_id contains both trainsg_dupes and test images that we do not want to leak 
    """
    community_name_to_img_id = defaultdict(set)
    print(" START *** exlude image length: ", len(exclude_img_id))
    all_img_id = set()
    exclude_img_id_local = set()
    exclude_img_id_local = exclude_img_id.copy()
    # print("***** split ****", split)
    ##################################
    # Iterate subject_str: e.g., cat
    ##################################
    for subject_str in dataset_scheme:        
        ##################################
        # Iterate community_name: e.g., cat(sofa)
        ##################################
        for community_name in dataset_scheme[subject_str]:
            ##################################
            # Iterate node_name: e.g., 'cat(cup)', 'cat(sofa)', 'cat(chair)'
            ##################################
            print("community name: ", community_name)
            for node_name in dataset_scheme[subject_str][community_name]:
                # print("node name: ", node_name)
                community_name_to_img_id[community_name].update(node_name_to_img_id[node_name] - exclude_img_id_local)
            # print("community_name_to_img_id first : ", len(community_name_to_img_id[community_name]))
            community_name_to_img_id[community_name] = community_name_to_img_id[community_name].intersection(include_img_id)
            # print("community_name_to_img_id after intersection: ", len(community_name_to_img_id[community_name]))
            community_name_to_img_id[community_name] = set(shuffle_and_truncate(community_name_to_img_id[community_name], trunc_size))
            print("community_name_to_img_id after truncation: ", len(community_name_to_img_id[community_name]))

            exclude_img_id_local.update(community_name_to_img_id[community_name])
            all_img_id.update(community_name_to_img_id[community_name])
        if max_additional_size > 0:
            for community_name in dataset_scheme[subject_str]:
                max_additional_size = int(max_additional_size)
                print("max_additional_size: ", max_additional_size)
                additional_size = int(max_additional_size)
                additional_imgs = set()
                print(" **** additional training data *** ")
                for node_name in additional_comms[subject_str]:
                    print("node name: ", node_name)
                    additional_imgs.update(node_name_to_img_id[node_name] - exclude_img_id_local)
                    print("additional image size: ", len(additional_imgs))
                    additional_imgs = additional_imgs.intersection(include_img_id)
                    print("additional image size after intersection with irm: ", len(additional_imgs))
                    additional_imgs = set(shuffle_and_truncate(additional_imgs, additional_size))
                    all_img_id.update(additional_imgs)
                    exclude_img_id_local.update(additional_imgs)
                    print("additional image size after truncation: ", len(additional_imgs))
                    if len(additional_imgs) < max_additional_size:
                        additional_size = max_additional_size - len(additional_imgs) 
                if len(additional_imgs) < max_additional_size:
                    ## take more data to fill in
                    ## remaining to add: 
                    rm_size = max_additional_size - len(additional_imgs) 
                    rm_imgs = include_img_id-all_img_id
                    community_name_to_img_id[community_name].update(set(shuffle_and_truncate(rm_imgs, rm_size)))
                    exclude_img_id_local.update(community_name_to_img_id[community_name])
                    all_img_id.update(community_name_to_img_id[community_name])

                community_name_to_img_id[community_name].update(additional_imgs)
                print("community_name_to_img_id after additions: ", len(community_name_to_img_id[community_name]))

        ##################################
        # Iterate community_name: e.g., cat(sofa)
        ##################################
        if copy:
            root_folder = os.path.join(dataset_folder, split)
            copy_image_for_subject(root_folder, subject_str, dataset_scheme[subject_str], community_name_to_img_id, trainsg_dupes=set(), use_symlink=False) # use False to share 

    return community_name_to_img_id, all_img_id

def parse_dataset_scheme_community_name(dataset_scheme, node_name_to_img_id, exclude_img_id=set(), split='test', copy=True, trunc_size=2500):
    """
    exclude_img_id contains both trainsg_dupes and test images that we do not want to leak 
    """
    community_name_to_img_id = defaultdict(set)
    all_img_id = set()

    ##################################
    # Iterate subject_str: e.g., cat
    ##################################
    for subject_str in dataset_scheme:        
        ##################################
        # Iterate community_name: e.g., cat(sofa)
        ##################################
        for community_name in dataset_scheme[subject_str]:
            ##################################
            # Iterate node_name: e.g., 'cat(cup)', 'cat(sofa)', 'cat(chair)'
            ##################################
            # for node_name in dataset_scheme[subject_str][community_name]:
                # print("node name: ", node_name)
            community_name_to_img_id[community_name].update(node_name_to_img_id[community_name] - exclude_img_id)
            all_img_id.update(node_name_to_img_id[community_name] - exclude_img_id)
            community_name_to_img_id[community_name] = set(shuffle_and_truncate(community_name_to_img_id[community_name], trunc_size))
            if copy:
                print(community_name, 'Size:', len(community_name_to_img_id[community_name]) )
            


        ##################################
        # Iterate community_name: e.g., cat(sofa)
        ##################################
        if copy:
            root_folder = os.path.join(CUSTOM_SPLIT_DATASET_FOLDER, split)
            copy_image_for_subject(root_folder, subject_str, dataset_scheme[subject_str], community_name_to_img_id, trainsg_dupes=set(), use_symlink=False) # use False to share 

    return community_name_to_img_id, all_img_id

def shuffle_and_truncate(img_id_set, truncate=2500):
        img_id_list = sorted(img_id_set)
        random.Random(42).shuffle(img_id_list)
        return img_id_list[:truncate]

def get_all_nodes_in_dataset(dataset_scheme):
    all_nodes = set()
    ##################################
    # Iterate subject_str: e.g., cat
    ##################################
    for subject_str in dataset_scheme:        
        ##################################
        # Iterate community_name: e.g., cat(sofa)
        ##################################
        for community_name in dataset_scheme[subject_str]:
            ##################################
            # Iterate node_name: e.g., 'cat(cup)', 'cat(sofa)', 'cat(chair)'
            ##################################
            for node_name in dataset_scheme[subject_str][community_name]:
                all_nodes.add(node_name)
    return all_nodes

def copy_images(root_folder,  split, subset_str, img_IDs, use_symlink=True):
    ##################################
    # Create dataset a new folder 
    ##################################
    subject_localgroup_folder = os.path.join(root_folder, split, subset_str)
    if os.path.isdir(subject_localgroup_folder): 
        shutil.rmtree(subject_localgroup_folder) 
    os.makedirs(subject_localgroup_folder, exist_ok = False)

    for image_idx_in_set, imageID in enumerate(img_IDs): 

        src_image_path = IMAGE_DATA_FOLDER + imageID + '.jpg'
        dst_image_path = os.path.join(subject_localgroup_folder, imageID + '.jpg') 

        if use_symlink:
            ##################################
            # Image Copy Option B: create symbolic link
            # Usage: for local use, saving disk storge. 
            ##################################
            os.symlink(src_image_path, dst_image_path)
            # print('symlink:', src_image_path, dst_image_path)
        else:
            ##################################
            # Image Copy Option A: copy the whole jpg file
            # Usage: for sharing the meta-dataset
            ################################## 
            shutil.copyfile(src_image_path, dst_image_path)
            # print('copy:', src_image_path, dst_image_path)

    return 

def generate_splitted_metadaset(args):
    CUSTOM_SPLIT_DATASET_FOLDER = args.dataset_name
    if os.path.isdir(CUSTOM_SPLIT_DATASET_FOLDER): 
        shutil.rmtree(CUSTOM_SPLIT_DATASET_FOLDER) 
    os.makedirs(CUSTOM_SPLIT_DATASET_FOLDER, exist_ok = False)


    node_name_to_img_id, most_common_list, subjects_to_all_set, subject_group_summary_dict = preprocess_groups(output_files_flag=False)

    ##################################
    # Removing ambiguous images that have both cats and dogs 
    ##################################
    trainsg_dupes = node_name_to_img_id['cat(dog)'] # can also use 'dog(cat)'
    subject_str_to_Graphs = dict()


    for subject_str in ['cat', 'dog']:
        subject_data = [ x for x in subject_group_summary_dict[subject_str].keys() if x not in ['cat(dog)', 'dog(cat)'] ]
        # print('subject_data', subject_data)
        ##################################
        # Print detected communities in Meta-Graph
        ##################################
        G = print_communities(subject_data, node_name_to_img_id, trainsg_dupes, subject_str) # print detected communities, which guides us the train/test split. 
        subject_str_to_Graphs[subject_str] = G


    
    test_set_scheme = {
        'cat': {
            'cat(shelf)': {'cat(container)', 'cat(shelf)', 'cat(vase)', 'cat(bowl)'},
        },
        'dog': {
            # In MetaDataset paper, the test images are all dogs. However, for completeness, we also provide cat images here. 
            'dog(shelf)': {'dog(desk)', 'dog(screen)', 'dog(laptop)', 'dog(shelf)', 'dog(picture)', 'dog(chair)'}, 
        },
    }

    additional_test_set_scheme = {
        'cat': {
            'cat(grass)': {'cat(house)', 'cat(car)', 'cat(grass)', 'cat(bird)'},
            'cat(sink)': {'cat(sink)', 'cat(bottle)', 'cat(faucet)', 'cat(towel)', 'cat(toilet)'}, 
            'cat(computer)': {'cat(speaker)', 'cat(computer)', 'cat(screen)', 'cat(laptop)', 'cat(computer mouse)', 'cat(keyboard)', 'cat(monitor)', 'cat(desk)',}, 
            'cat(box)': {'cat(box)', 'cat(paper)', 'cat(suitcase)', 'cat(bag)',}, 
            # 'cat(book)': {'cat(books)', 'cat(book)', 'cat(television)', 'cat(bookshelf)', 'cat(blinds)',},
        },
        'dog': {
            'dog(sofa)': {'dog(sofa)', 'dog(television)', 'dog(carpet)',  'dog(phone)', 'dog(book)',}, 
            'dog(grass)': {'dog(house)', 'dog(grass)', 'dog(horse)', 'dog(cow)', 'dog(sheep)','dog(animal)'}, 
            'dog(vehicle)': {'dog(car)', 'dog(motorcycle)', 'dog(truck)', 'dog(bike)', 'dog(basket)', 'dog(bicycle)', 'dog(skateboard)', }, 
            # 'dog(cap)': {'dog(cap)', 'dog(scarf)', 'dog(jacket)', 'dog(toy)', 'dog(collar)', 'dog(tie)'},
        },
    }



    print('========== test set info ==========')

    test_size = 100 #80
    # sub_train_size = 60 #120
    sub_train_size = 100
    val_sub_size = 40
    add_p = args.add_p
    # add_p = 0.25 #0, 0.10, 0.25

    dataset_folder = CUSTOM_SPLIT_DATASET_FOLDER+'/p1'
    test_community_name_to_img_id, test_all_img_id = parse_dataset_scheme(test_set_scheme, node_name_to_img_id,dataset_folder=dataset_folder, exclude_img_id=trainsg_dupes, split='test',trunc_size=test_size)
    
    dataset_folder = CUSTOM_SPLIT_DATASET_FOLDER+'/p2'
    test_community_name_to_img_id, test_all_img_id = parse_dataset_scheme(test_set_scheme, node_name_to_img_id,dataset_folder=dataset_folder, exclude_img_id=trainsg_dupes, split='test',trunc_size=test_size)

    dataset_folder = CUSTOM_SPLIT_DATASET_FOLDER+'/irm'
    test_community_name_to_img_id, test_all_img_id = parse_dataset_scheme(test_set_scheme, node_name_to_img_id,dataset_folder=dataset_folder, exclude_img_id=trainsg_dupes, split='test',trunc_size=test_size)

    
    print("============ Metashift IRM ================")
    train_set_scheme = {
        # Note: these comes from copy-pasting the community detection results of cat & dog. 
        'cat': {
            # The cat training data is always cat(\emph{sofa + bed}) 
            'cat(P1)': {'cat(cup)', 'cat(sofa)', 'cat(chair)'},
            'cat(P2)':  {'cat(bed)', 'cat(comforter)', 'cat(sheet)', 'cat(blanket)', 'cat(remote control)', 'cat(pillow)', 'cat(couch)'},
        }, 
        'dog': {
            # Experiment 1: the dog training data is dog(\emph{cabinet + bed}) communities, and its distance to dog(\emph{shelf}) is $d$=0.44. 
            'dog(P1)': {'dog(floor)', 'dog(clothes)', 'dog(towel)', 'dog(door)', 'dog(rug)', 'dog(cabinet)'}, 
            'dog(P2)': {'dog(blanket)', 'dog(bed)', 'dog(sheet)', 'dog(remote control)', 'dog(pillow)', 'dog(lamp)', 'dog(couch)', 'dog(books)', 'dog(curtain)'}
        
            # Experiment 2: 
            # 'dog(P1)': {'dog(bag)', 'dog(backpack)', 'dog(purse)','dog(suitcase)','dog(jacket)'},
            # 'dog(P2)': {'dog(box)', 'dog(container)', 'dog(food)', 'dog(table)', 'dog(plate)', 'dog(cup)','dog(basket)','dog(pole)'} ,
            
            # Experiment 3: the dog training data is dog(\emph{bench + bike}) with distance $d$=1.12
            # 'dog(P1)': {'dog(bench)', 'dog(trash can)','dog(fence)','dog(trees)','dog(frisbee)','dog(truck)'} ,
            # 'dog(P2)': {'dog(basket)', 'dog(woman)', 'dog(bike)', 'dog(bicycle)','dog(car)','dog(bottle)'},

            # Experiment 4: the dog training data is dog(\emph{boat + surfboard}) with distance $d$=1.43.   
            # 'dog(P1)': {'dog(frisbee)', 'dog(rope)', 'dog(flag)', 'dog(trees)', 'dog(boat)','dog(dirt)'},
            # 'dog(P2)': {'dog(water)', 'dog(surfboard)', 'dog(sand)', 'dog(ball)','dog(cap)','dog(shirt)','dog(glasses)'}
        
        }
    }
    val_set_scheme = {
        # Note: these comes from copy-pasting the community detection results of cat & dog. 
        'cat': {
            # The cat training data is always cat(\emph{sofa + bed}) 
            'cat(P1)_val': {'cat(cup)', 'cat(sofa)', 'cat(chair)'},
            'cat(P2)_val':  {'cat(bed)', 'cat(comforter)', 'cat(sheet)', 'cat(blanket)', 'cat(remote control)', 'cat(pillow)', 'cat(couch)'},
        }, 
        'dog': {
            # Experiment 1: the dog training data is dog(\emph{cabinet + bed}) communities, and its distance to dog(\emph{shelf}) is $d$=0.44. 
            # 'dog(P1)_val': {'dog(floor)', 'dog(clothes)', 'dog(towel)', 'dog(door)', 'dog(rug)', 'dog(cabinet)'}, 
            # 'dog(P2)_val': {'dog(blanket)', 'dog(bed)', 'dog(sheet)', 'dog(remote control)', 'dog(pillow)', 'dog(lamp)', 'dog(couch)', 'dog(books)', 'dog(curtain)'}

             # Experiment 2: 
            # 'dog(P1)_val':  {'dog(bag)', 'dog(backpack)', 'dog(purse)','dog(suitcase)','dog(jacket)'},
            # 'dog(P2)_val': {'dog(box)', 'dog(container)', 'dog(food)', 'dog(table)', 'dog(plate)', 'dog(cup)','dog(basket)','dog(pole)'} 

            # Experiment 3: the dog training data is dog(\emph{bench + bike}) with distance $d$=1.12
            # 'dog(P1)_val': {'dog(bench)', 'dog(trash can)','dog(fence)','dog(trees)','dog(frisbee)','dog(truck)'} ,
            # 'dog(P2)_val': {'dog(basket)', 'dog(woman)', 'dog(bike)', 'dog(bicycle)','dog(car)','dog(bottle)'}

            # Experiment 4: the dog training data is dog(\emph{boat + surfboard}) with distance $d$=1.43.   
            'dog(P1)_val': {'dog(frisbee)', 'dog(rope)', 'dog(flag)', 'dog(trees)', 'dog(boat)','dog(dirt)'},
            'dog(P2)_val': {'dog(water)', 'dog(surfboard)', 'dog(sand)', 'dog(ball)','dog(cap)','dog(shirt)','dog(glasses)'}
        }
    }

    dataset_folder = CUSTOM_SPLIT_DATASET_FOLDER+'/irm'
    os.makedirs(os.path.dirname(dataset_folder + '/' + 'imageID_to_group.pkl'), exist_ok=True)
    
    train_community_name_to_img_id, train_all_img_id = parse_dataset_scheme(train_set_scheme,node_name_to_img_id,dataset_folder=dataset_folder, exclude_img_id=test_all_img_id.union(trainsg_dupes), split='train',trunc_size=sub_train_size)
    val_community_name_to_img_id, val_all_img_id = parse_dataset_scheme(val_set_scheme,node_name_to_img_id,dataset_folder=dataset_folder, exclude_img_id=test_all_img_id.union(trainsg_dupes).union(train_all_img_id), split='val_out_of_domain',trunc_size=val_sub_size)

    with open(dataset_folder + '/' + 'imageID_to_group.pkl', 'wb') as handle:
        imageID_to_group = dict()
        group_to_imageID = train_community_name_to_img_id.copy()
        group_to_imageID.update(test_community_name_to_img_id)
        group_to_imageID.update(val_community_name_to_img_id)
        for group_str in group_to_imageID:
            print('group string: ', group_str,' size: ', len(group_to_imageID[group_str]))
            for imageID in group_to_imageID[group_str]:
                if imageID not in imageID_to_group:
                    imageID_to_group[imageID] = [group_str] 
                else:
                    imageID_to_group[imageID].append(group_str)
        pickle.dump(imageID_to_group, handle)
        print('end dumping ### ')
    print("### File exists? ")
    print(os.path.exists(dataset_folder + '/' + 'imageID_to_group.pkl'))
    print(dataset_folder + '/' + 'imageID_to_group.pkl')
    

    print('========== Quantifying the distance between train and test subsets ==========')
    test_community_name_to_img_id_all, _ = parse_dataset_scheme(test_set_scheme, node_name_to_img_id, exclude_img_id=trainsg_dupes, split='test', copy=False)
    # train_community_name_to_img_id, _ = parse_dataset_scheme(train_set_scheme_all, node_name_to_img_id, exclude_img_id=trainsg_dupes, split='train', copy=False)
    additional_test_community_name_to_img_id, _ = parse_dataset_scheme(additional_test_set_scheme, node_name_to_img_id, exclude_img_id=trainsg_dupes, split='test', copy=False)
    
    # exit(0)
    community_name_to_img_id = test_community_name_to_img_id_all.copy()
    community_name_to_img_id.update(train_community_name_to_img_id)
    # community_name_to_img_id.update(val_community_name_to_img_id)
    print("community_name_to_img_id.keys()", community_name_to_img_id.keys())
    community_name_to_img_id.update(additional_test_community_name_to_img_id)
    dog_community_name_list = sorted(train_set_scheme['dog']['dog(P1)']) +sorted(train_set_scheme['dog']['dog(P2)'])+ sorted(test_set_scheme['dog']) + sorted(additional_test_set_scheme['dog']) 
    dog_community_name_list = sorted(train_set_scheme['dog'])+ sorted(test_set_scheme['dog']) + sorted(additional_test_set_scheme['dog']) 
    N_sets = len(community_name_to_img_id.keys())
    # N_sets = len(dog_community_name_list)
    Adjacency_matrix = np.ones((N_sets, N_sets))
    for i, ii in enumerate(community_name_to_img_id.keys()):
        
        for j,jj in enumerate(community_name_to_img_id.keys()):
            set_A = community_name_to_img_id[ii] - trainsg_dupes
            set_B = community_name_to_img_id[jj] - trainsg_dupes
            overlap_set = set_A.intersection(set_B)
  
            if len(overlap_set) == 0:
                edge_weight = 0
            else: 
                edge_weight = len(overlap_set) / min( len(set_A), len(set_B) )
            Adjacency_matrix[i,j] = Adjacency_matrix[j,i] = edge_weight
            
    # G, pars_sc, Adjacency_matrix = build_subset_graph(dog_community_name_list, community_name_to_img_id, trainsg_dupes=set(), subject_str=None,seed=0)
    # G_dog_all, dog_pars_sc, Adjacency_matrix = build_subset_graph(dog_community_name_list, community_name_to_img_id, trainsg_dupes=set(), subject_str=None,seed=0)
    labels = []
    for i, x in enumerate(community_name_to_img_id.keys()):
    # for i, x in enumerate(dog_community_name_list):
        # add a \n
        labels.append(x.replace('(', '\n('))
    A_pd = pd.DataFrame(np.matrix(Adjacency_matrix), index=labels, columns=labels)
    G_dog_all = nx.from_pandas_adjacency(A_pd)
    print("Nodes: ", G_dog_all.nodes())
    spectral_pos = nx.spectral_layout(
        G=G_dog_all, 
        dim=5,
        )
    
    dists = []
    subs = []
    
    for sub in train_set_scheme['dog']:
        subs.append(sub)
        dists.append(np.linalg.norm(spectral_pos[sub.replace('(','\n(')] - spectral_pos['dog\n(shelf)']))        
    
    print('Distance from {}+{} to {}: {}'.format(
            subs[0], subs[1], 'dog(shelf)', 
            0.5 * (dists[0] + dists[1])
            ))
    print(f"Distance from {subs[0]} is {dists[0]}\nDistance from {subs[1]} is {dists[1]} ")
    min_dist = np.argmin(dists)
    min_partition = subs[min_dist]

    max_dist = np.argmax(dists)
    max_partition = subs[max_dist]

    print(f"minimum-distant partition {min_partition} with distance of {dists[min_dist]} with communities: ")
    print("\t",train_set_scheme['dog'][min_partition])
    
    print()
    print()
    print("=============== ADDITIONAL DATA SAMPLES FROM SECOND ENV ==============")
    test_community_name_to_img_id_all, _ = parse_dataset_scheme(test_set_scheme, node_name_to_img_id, exclude_img_id=trainsg_dupes, split='test', copy=False)
    # train_community_name_to_img_id, _ = parse_dataset_scheme(train_set_scheme_all, node_name_to_img_id, exclude_img_id=trainsg_dupes, split='train', copy=False)
    additional_test_community_name_to_img_id, _ = parse_dataset_scheme(additional_test_set_scheme, node_name_to_img_id, exclude_img_id=trainsg_dupes, split='test', copy=False)

    community_name_to_img_id = test_community_name_to_img_id_all.copy()
    community_name_to_img_id.update(train_community_name_to_img_id)
    # community_name_to_img_id.update(val_community_name_to_img_id)
    community_name_to_img_id.update(additional_test_community_name_to_img_id)    
    
    for p_com in train_set_scheme['dog'][max_partition]:
        community_name_to_img_id.update({p_com:node_name_to_img_id[p_com]})
    
    for p_com in train_set_scheme['cat'][f"cat({max_partition.split('(')[1]}"]:
        community_name_to_img_id.update({p_com:node_name_to_img_id[p_com]})

    N_sets = len(community_name_to_img_id.keys())
    # N_sets = len(dog_community_name_list)
    Adjacency_matrix = np.ones((N_sets, N_sets))
    for i, ii in enumerate(community_name_to_img_id.keys()):
    # for i, ii in enumerate(dog_community_name_list):
        # print("ii: ",ii)
        
        for j,jj in enumerate(community_name_to_img_id.keys()):
        # for j,jj in enumerate(dog_community_name_list):
            set_A = community_name_to_img_id[ii] - trainsg_dupes
            set_B = community_name_to_img_id[jj] - trainsg_dupes
            overlap_set = set_A.intersection(set_B)
  
            if len(overlap_set) == 0:
                edge_weight = 0
            else: 
                edge_weight = len(overlap_set) / min( len(set_A), len(set_B) )
            Adjacency_matrix[i,j] = Adjacency_matrix[j,i] = edge_weight
            
    labels = []
    for i, x in enumerate(community_name_to_img_id.keys()):
    # for i, x in enumerate(dog_community_name_list):
        # add a \n
        labels.append(x.replace('(', '\n('))
    A_pd = pd.DataFrame(np.matrix(Adjacency_matrix), index=labels, columns=labels)
    G_dog_all = nx.from_pandas_adjacency(A_pd)
    print("Nodes: ", G_dog_all.nodes())
    spectral_pos = nx.spectral_layout(
        G=G_dog_all, 
        dim=5,
        # dim = (3,2)
        )
    
    dists = []
    subs = []
    
    for sub in train_set_scheme['dog'][max_partition]:
        subs.append(sub)
        dist = np.linalg.norm(spectral_pos[sub.replace('(','\n(')] - spectral_pos['dog\n(shelf)'])
        dists.append(dist)        
    subs = np.array(subs)
    dog_sorted_additional_comms = subs[np.argsort(dists)]

    dists = []
    subs = []
    
    for sub in train_set_scheme['cat'][f"cat({max_partition.split('(')[1]}"]:
        subs.append(sub)
        dist = np.linalg.norm(spectral_pos[sub.replace('(','\n(')] - spectral_pos['cat\n(shelf)'])
        dists.append(dist)        
    subs = np.array(subs)
    cat_sorted_additional_comms = subs[np.argsort(dists)]

    # print("cat_sorted_additional_comms: ", cat_sorted_additional_comms)
    print("============ spectral clustering partitioning for ==============")
    
    com_to_img_id = {}
    exclude_img_id=test_all_img_id.union(trainsg_dupes)
    
    # cat_sub_coms = list(cat_sorted_additional_comms)
    # dog_sub_coms = list(dog_sorted_additional_comms)

    cat_sub_coms = []
    dog_sub_coms = []
    for com_i in train_set_scheme['dog'][min_partition]:
        dog_sub_coms.append(com_i)
    
    for com_i in train_set_scheme['cat'][f"cat({min_partition.split('(')[1]}"]:
        cat_sub_coms.append(com_i)

    G, dog_clusters, adj_training = build_subset_graph(dog_sub_coms, node_name_to_img_id, trainsg_dupes=trainsg_dupes, subject_str=None, seed=0)
    G, cat_clusters, adj_training = build_subset_graph(cat_sub_coms, node_name_to_img_id, trainsg_dupes=trainsg_dupes, subject_str=None, seed=0)

     
    print("cat communities: ", cat_clusters)
    print("dog communities: ", dog_clusters)

    cat_clusters[0] = [i.replace('\n(','(') for i in cat_clusters[0]]
    cat_clusters[1] = [i.replace('\n(','(') for i in cat_clusters[1]]

    dog_clusters[0] = [i.replace('\n(','(') for i in dog_clusters[0]]
    dog_clusters[1] = [i.replace('\n(','(') for i in dog_clusters[1]]

    print("cat communities: ", cat_clusters)
    print("dog communities: ", dog_clusters)

    ### writing down the new partitions of the closer cluster

    train_set_scheme_P1 = {'dog':{
        'dog(P11)': dog_clusters[0],
        'dog(P12)': dog_clusters[1]
        },
    'cat':{
        'cat(P11)': cat_clusters[0],
        'cat(P12)': cat_clusters[1]
            }}
    val_set_scheme_P1 = {'dog':{
        'dog(P11)_val': dog_clusters[0],
        'dog(P12)_val': dog_clusters[1]
        },
    'cat':{
        'cat(P11)_val': cat_clusters[0],
        'cat(P12)_val': cat_clusters[1]
            }}
    
    additional_comms = {"dog":dog_sorted_additional_comms, 
    "cat":cat_sorted_additional_comms}
    
    dataset_folder = CUSTOM_SPLIT_DATASET_FOLDER+'/p1'
    train_community_name_to_img_id_P1, train_all_img_id_P1 = parse_dataset_scheme_preexisting_data(train_set_scheme_P1, node_name_to_img_id,dataset_folder=dataset_folder, exclude_img_id=test_all_img_id.union(trainsg_dupes), include_img_id=train_all_img_id, additional_comms=additional_comms,max_additional_size=(sub_train_size/2)*add_p, split='train',trunc_size=int(sub_train_size/2))
    val_community_name_to_img_id_P1, val_all_img_id_P1 = parse_dataset_scheme_preexisting_data(val_set_scheme_P1, node_name_to_img_id,dataset_folder=dataset_folder, exclude_img_id=test_all_img_id.union(trainsg_dupes).union(train_all_img_id_P1), include_img_id=val_all_img_id, additional_comms= additional_comms,max_additional_size=(val_sub_size/2)*add_p, split='val_out_of_domain',trunc_size=int(val_sub_size/2))
    
        
    os.makedirs(os.path.dirname(dataset_folder + '/' + 'imageID_to_group.pkl'), exist_ok=True)
    
    with open(dataset_folder + '/' + 'imageID_to_group.pkl', 'wb') as handle:
        imageID_to_group = dict()
        group_to_imageID = train_community_name_to_img_id_P1.copy()
        group_to_imageID.update(test_community_name_to_img_id)
        group_to_imageID.update(val_community_name_to_img_id_P1)
        for group_str in group_to_imageID:
            print('group string: ', group_str,' size: ', len(group_to_imageID[group_str]))
            for imageID in group_to_imageID[group_str]:
                if imageID not in imageID_to_group:
                    imageID_to_group[imageID] = [group_str] 
                else:
                    imageID_to_group[imageID].append(group_str)
        pickle.dump(imageID_to_group, handle)
        print('end dumping ### ')
    
    for sub_str in train_community_name_to_img_id_P1:
        sub_a = sub_str.split('(')[0]
        dst = sub_a+'/'+sub_str
        copy_images(
            dataset_folder, 'train', dst, use_symlink=False,
            img_IDs =  train_community_name_to_img_id_P1[sub_str]
            )
    for sub_str in val_community_name_to_img_id_P1:
        sub_a = sub_str.split('(')[0]
        dst = sub_a+'/'+sub_str
        copy_images(
            dataset_folder, 'val_out_of_domain', dst, use_symlink=False,
            img_IDs =  val_community_name_to_img_id_P1[sub_str]
            )    
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Domain Generalization dataset')
    parser.add_argument('--dataset_name', type=str, default='data/MetaShift/MetaShift-domain-generalization-exp')
    parser.add_argument('--add_p', type=float, default=0, help='used values are {0, 0.10, 0.25}')
    args = parser.parse_args()
    generate_splitted_metadaset(args)
