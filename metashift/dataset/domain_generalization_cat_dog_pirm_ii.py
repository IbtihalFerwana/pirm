"""
Generate MetaDataset with train/test split 
"""

CUSTOM_SPLIT_DATASET_FOLDER = 'data/MetaShift/Domain-Generalization-Cat-Dog-pirmii-exp1'

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


def parse_dataset_scheme_preexisting_data(dataset_scheme, node_name_to_img_id,dataset_folder=None, exclude_img_id=set(),include_img_id=set(), split='test', copy=True, trunc_size=2500):
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
                community_name_to_img_id[community_name].update(node_name_to_img_id[node_name] - exclude_img_id_local)
            # print("community_name_to_img_id first : ", len(community_name_to_img_id[community_name]))
            community_name_to_img_id[community_name] = community_name_to_img_id[community_name].intersection(include_img_id)
            # print("community_name_to_img_id after intersection: ", len(community_name_to_img_id[community_name]))
            community_name_to_img_id[community_name] = set(shuffle_and_truncate(community_name_to_img_id[community_name], trunc_size))
            # print("community_name_to_img_id after truncation: ", len(community_name_to_img_id[community_name]))

            exclude_img_id_local.update(community_name_to_img_id[community_name])
            all_img_id.update(community_name_to_img_id[community_name])
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

def generate_splitted_metadaset():

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


    train_set_scheme_pirm = {
        # Note: these comes from copy-pasting the community detection results of cat & dog. 
        'cat': [
            # The cat training data is always cat(\emph{sofa + bed}) 
            'cat(cup)', 'cat(sofa)', 'cat(chair)',
            'cat(bed)', 'cat(comforter)', 'cat(sheet)', 'cat(blanket)', 'cat(remote control)', 'cat(pillow)', 'cat(couch)'],
        'dog': [
             # Experiment 1: the dog training data is dog(\emph{cabinet + bed}) communities, and its distance to dog(\emph{shelf}) is $d$=0.44. 
            'dog(floor)', 'dog(clothes)', 'dog(towel)', 'dog(door)', 'dog(rug)', 'dog(cabinet)', 
            'dog(blanket)', 'dog(bed)', 'dog(sheet)', 'dog(remote control)', 'dog(pillow)', 'dog(lamp)', 'dog(couch)', 'dog(books)', 'dog(curtain)' ]
        # 'dog': [
            #   Experiment 1: the dog training data is dog(\emph{cabinet + bed}) communities, and its distance to dog(\emph{shelf}) is $d$=0.44. 
            # 'dog(shelf)', 'dog(computer)', 'dog(bed)', 'dog(sheet)', 'dog(remote control)', 'dog(desk)', 'dog(lamp)', 'dog(couch)', 'dog(sofa)', 'dog(television)' ]
           
            # # Experiment 2: the dog training data is dog(\emph{bag + box}), and its distance to dog(\emph{shelf}) is $d$=0.71. 
            # 'dog': ['dog(bag)', 'dog(backpack)', 'dog(purse)','dog(water)',
            # 'dog(box)', 'dog(container)', 'dog(food)', 'dog(table)', 'dog(plate)', 'dog(cup
            # Experiment 3: the dog training data is dog(\emph{bench + bike}) with distance $d$=1.12
            # 'dog':['dog(bench)', 'dog(trash can)' ,'dog(basket)', 'dog(woman)', 'dog(bike)', 'dog(bicycle)']

            # Experiment 4: the dog training data is dog(\emph{boat + surfboard}) with distance $d$=1.43.   
            # 'dog':['dog(frisbee)', 'dog(rope)', 'dog(flag)', 'dog(trees)', 'dog(boat)', 'dog(water)', 'dog(surfboard)', 'dog(sand)','dog(ball)','dog(grass)','dog(boy)','dog(dirt)']
        
    }
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
    val_sub_size = 50

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
            'dog(P1)_val': {'dog(floor)', 'dog(clothes)', 'dog(towel)', 'dog(door)', 'dog(rug)', 'dog(cabinet)'}, 
            'dog(P2)_val': {'dog(blanket)', 'dog(bed)', 'dog(sheet)', 'dog(remote control)', 'dog(pillow)', 'dog(lamp)', 'dog(couch)', 'dog(books)', 'dog(curtain)'}
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
    N_sets = len(community_name_to_img_id.keys())
    Adjacency_matrix = np.ones((N_sets, N_sets))
    for i, ii in enumerate(community_name_to_img_id.keys()):
        # print("ii: ",ii)
        
        for j,jj in enumerate(community_name_to_img_id.keys()):
            set_A = community_name_to_img_id[ii] - trainsg_dupes
            set_B = community_name_to_img_id[jj] - trainsg_dupes
            overlap_set = set_A.intersection(set_B)
            if ii == 'dog(shelf)':
                print('jj:',jj)
                print("overlap set length: ", len(overlap_set))
                print("length ii: ", len(set_A), " length jj: ",len(set_B))
                print('**')
            if len(overlap_set) == 0:
                edge_weight = 0
            else: 
                edge_weight = len(overlap_set) / min( len(set_A), len(set_B) )
            Adjacency_matrix[i,j] = Adjacency_matrix[j,i] = edge_weight
            
    # G, pars_sc, Adjacency_matrix = build_subset_graph(dog_community_name_list, community_name_to_img_id, trainsg_dupes=set(), subject_str=None,seed=0)
    # G_dog_all, dog_pars_sc, Adjacency_matrix = build_subset_graph(dog_community_name_list, community_name_to_img_id, trainsg_dupes=set(), subject_str=None,seed=0)
    labels = []
    for i, x in enumerate(community_name_to_img_id.keys()):
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
    print(f"minimum-distant partition {min_partition} with distance of {dists[min_dist]} with communities: ")
    print("\t",train_set_scheme['dog'][min_partition])

    print("============ spectral clustering partitioning for ==============")
    
    com_to_img_id = {}
    exclude_img_id=test_all_img_id.union(trainsg_dupes)
    print("exclude images length: ", len(exclude_img_id))
    
    cat_sub_coms = []
    dog_sub_coms = []
    for com_i in train_set_scheme['dog'][min_partition]:
        com_i = com_i.split('(')[-1][:-1]
        dog_sub_coms.append(com_i)
    for com_i in train_set_scheme['cat'][f"cat({min_partition.split('(')[1]}"]:
        com_i = com_i.split('(')[-1][:-1]
        cat_sub_coms.append(com_i)

    G, dog_clusters, adj_training = build_subset_graph(dog_sub_coms, node_name_to_img_id, trainsg_dupes=trainsg_dupes, subject_str=None, seed=0)
    G, cat_clusters, adj_training = build_subset_graph(cat_sub_coms, node_name_to_img_id, trainsg_dupes=trainsg_dupes, subject_str=None, seed=0)
    print("cat communities: ", cat_clusters)
    print("dog communities: ", dog_clusters)


    ### writing down the new partitions of the closer cluster

    train_set_scheme_P1 = {'dog':{
        'dog(P11)':set([ f'dog({x})' for x in dog_clusters[0]]),
        'dog(P12)':set([ f'dog({x})' for x in dog_clusters[1]])
        },
    'cat':{
        'cat(P11)':set([ f'cat({x})'  for x in cat_clusters[0]]),
        'cat(P12)':set([ f'cat({x})'  for x in cat_clusters[1]])
            }}
    val_set_scheme_P1 = {'dog':{
        'dog(P11)_val':set([ f'dog({x})' for x in dog_clusters[0]]),
        'dog(P12)_val':set([ f'dog({x})' for x in dog_clusters[1]])
        },
    'cat':{
        'cat(P11)_val':set([ f'cat({x})'  for x in cat_clusters[0]]),
        'cat(P12)_val':set([ f'cat({x})'  for x in cat_clusters[1]])
            }}
    
    dataset_folder = CUSTOM_SPLIT_DATASET_FOLDER+'/p1'
    train_community_name_to_img_id_P1, train_all_img_id_P1 = parse_dataset_scheme_preexisting_data(train_set_scheme_P1, node_name_to_img_id,dataset_folder=dataset_folder, exclude_img_id=test_all_img_id.union(trainsg_dupes), include_img_id=train_all_img_id, split='train',trunc_size=int(sub_train_size/2))
    val_community_name_to_img_id_P1, val_all_img_id_P1 = parse_dataset_scheme_preexisting_data(val_set_scheme_P1, node_name_to_img_id,dataset_folder=dataset_folder, exclude_img_id=test_all_img_id.union(trainsg_dupes).union(train_all_img_id_P1), include_img_id=val_all_img_id, split='val_out_of_domain',trunc_size=int(val_sub_size/2))
    
        
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


    exit(0) 
    
    print("============ spectral clustering partitioning ==============")
    # train_community_name_to_img_id, train_all_img_id = parse_dataset_scheme_community_name(train_set_scheme, node_name_to_img_id, exclude_img_id=test_all_img_id.union(trainsg_dupes), split='train', copy=False)
    com_to_img_id = {}
    exclude_img_id=test_all_img_id.union(trainsg_dupes)
    print("exclude images length: ", len(exclude_img_id))
    common_sub_communities = set()
    cat_sub_coms = []
    dog_sub_coms = []
    # for a in ['cat','dog']:
    #     for com in train_set_scheme_pirm[a]:
    #         com_i = com.split('(')[-1][:-1]
    #         common_sub_communities.add(com_i)
    
    for com_i in train_set_scheme_pirm['dog']:
        com_i = com_i.split('(')[-1][:-1]
        dog_sub_coms.append(com_i)
    for com_i in train_set_scheme_pirm['cat']:
        com_i = com_i.split('(')[-1][:-1]
        cat_sub_coms.append(com_i)
    # print("communities: ",common_sub_communities)

    G, dog_clusters, adj_training = build_subset_graph(dog_sub_coms, train_community_name_to_img_id, trainsg_dupes=trainsg_dupes, subject_str=None, seed=0)
    G, cat_clusters, adj_training = build_subset_graph(cat_sub_coms, train_community_name_to_img_id, trainsg_dupes=trainsg_dupes, subject_str=None, seed=0)
    print("cat communities: ", cat_clusters)
    print("dog communities: ", dog_clusters)
    train_set_scheme_P1 = {'dog':{
        'dog(P11)':set([ f'dog({x})' for x in dog_clusters[0]]),
        'dog(P12)':set([ f'dog({x})' for x in dog_clusters[1]])
        },
    'cat':{
        'cat(P11)':set([ f'cat({x})'  for x in cat_clusters[0]]),
        'cat(P12)':set([ f'cat({x})'  for x in cat_clusters[1]])
            }}
    val_set_scheme_P1 = {'dog':{
        'dog(P11)_val':set([ f'dog({x})' for x in dog_clusters[0]]),
        'dog(P12)_val':set([ f'dog({x})' for x in dog_clusters[1]])
        },
    'cat':{
        'cat(P11)_val':set([ f'cat({x})'  for x in cat_clusters[0]]),
        'cat(P12)_val':set([ f'cat({x})'  for x in cat_clusters[1]])
            }}
    
    dataset_folder = CUSTOM_SPLIT_DATASET_FOLDER+'/p1'
    train_community_name_to_img_id_P1, train_all_img_id_P1 = parse_dataset_scheme_preexisting_data(train_set_scheme_P1, node_name_to_img_id,dataset_folder=dataset_folder, exclude_img_id=test_all_img_id.union(trainsg_dupes), include_img_id=train_all_img_id, split='train',trunc_size=sub_train_size)
    val_community_name_to_img_id_P1, val_all_img_id_P1 = parse_dataset_scheme_preexisting_data(val_set_scheme_P1, node_name_to_img_id,dataset_folder=dataset_folder, exclude_img_id=test_all_img_id.union(trainsg_dupes).union(train_all_img_id_P1), include_img_id=val_all_img_id, split='val_out_of_domain',trunc_size=val_sub_size)
    
    # print("train_community_name_to_img_id keys: ",train_community_name_to_img_id_P1.keys())
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
    
    for sub_str in train_community_name_to_img_id:
        sub_a = sub_str.split('(')[0]
        dst = sub_a+'/'+sub_str
        copy_images(
            dataset_folder, 'train', dst, use_symlink=False,
            img_IDs =  train_community_name_to_img_id[sub_str]
            )
    for sub_str in val_community_name_to_img_id:
        sub_a = sub_str.split('(')[0]
        dst = sub_a+'/'+sub_str
        copy_images(
            dataset_folder, 'val_out_of_domain', dst, use_symlink=False,
            img_IDs =  val_community_name_to_img_id[sub_str]
            )

    exit(0)
    G, par_sc_1, Adjacency_matrix = build_subset_graph(animals_clusters[0], com_to_img_id_1, trainsg_dupes=set(), subject_str=None,seed=0)
    G, par_sc_2, Adjacency_matrix = build_subset_graph(animals_clusters[1], com_to_img_id_2, trainsg_dupes=set(), subject_str=None,seed=0)
    print("======== clusters ========")
    print(par_sc_1)
    print(par_sc_2)

    train_set_scheme_P1 = {'dog':{
        'dog(P11)':set([ f'dog({x})' for x in par_sc_1[0]]),
        'dog(P12)':set([ f'dog({x})' for x in par_sc_1[1]])
        },
    'cat':{
        'cat(P11)':set([ f'cat({x})'  for x in par_sc_1[0]]),
        'cat(P12)':set([ f'cat({x})'  for x in par_sc_1[1]])
            }}
    val_set_scheme_P1 = {'dog':{
        'dog(P11)_val':set([ f'dog({x})' for x in par_sc_1[0]]),
        'dog(P12)_val':set([ f'dog({x})' for x in par_sc_1[1]])
        },
    'cat':{
        'cat(P11)_val':set([ f'cat({x})'  for x in par_sc_1[0]]),
        'cat(P12)_val':set([ f'cat({x})'  for x in par_sc_1[1]])
            }}
    com_to_img_id_1 = {}
    for a in par_sc_1:
        for com_i in a:
            cat_com = f'cat({com_i})'
            dog_com = f'dog({com_i})'
            cat_sub = node_name_to_img_id[cat_com]-exclude_img_id
            dog_sub = node_name_to_img_id[dog_com]-exclude_img_id
            com_to_img_id_1[f'cat({com_i})'] = cat_sub
            com_to_img_id_1[f'dog({com_i})'] = dog_sub

    
    train_set_scheme_P2 = {'dog':{
        'dog(P21)':set([ f'dog({x})' for x in par_sc_2[0]]),
        'dog(P22)':set([ f'dog({x})' for x in par_sc_2[1]])
        },
    'cat':{
        'cat(P21)':set([ f'cat({x})' for x in par_sc_2[0]]),
        'cat(P22)':set([ f'cat({x})' for x in par_sc_2[1]])
            }}
    
    val_set_scheme_P2 = {'dog':{
        'dog(P21)_val':set([ f'dog({x})' for x in par_sc_2[0]]),
        'dog(P22)_val':set([ f'dog({x})' for x in par_sc_2[1]])
        },
    'cat':{
        'cat(P21)_val':set([ f'cat({x})' for x in par_sc_2[0]]),
        'cat(P22)_val':set([ f'cat({x})' for x in par_sc_2[1]])
            }}

    com_to_img_id_2 = {}
    for a in par_sc_2:
        for com_i in a:
            cat_com = f'cat({com_i})'
            dog_com = f'dog({com_i})'
            cat_sub = node_name_to_img_id[cat_com]-exclude_img_id
            dog_sub = node_name_to_img_id[dog_com]-exclude_img_id
            com_to_img_id_2[f'cat({com_i})'] = cat_sub
            com_to_img_id_2[f'dog({com_i})'] = dog_sub

    dataset_folder = CUSTOM_SPLIT_DATASET_FOLDER+'/p1'
    train_community_name_to_img_id_P1, train_all_img_id_P1 = parse_dataset_scheme(train_set_scheme_P1, com_to_img_id_1,dataset_folder=dataset_folder, exclude_img_id=exclude_img_id, split='train',trunc_size=sub_train_size)
    val_community_name_to_img_id_P1, val_all_img_id_P1 = parse_dataset_scheme(val_set_scheme_P1, com_to_img_id_1,dataset_folder=dataset_folder, exclude_img_id=train_all_img_id_P1, split='val_out_of_domain',trunc_size=val_sub_size)

    print("train_community_name_to_img_id keys: ",train_community_name_to_img_id_P1.keys())
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
    
    print(os.path.exists(dataset_folder + '/' + 'imageID_to_group.pkl'))
    print(dataset_folder + '/' + 'imageID_to_group.pkl')
    
    dataset_folder = CUSTOM_SPLIT_DATASET_FOLDER+'/p2'
    train_community_name_to_img_id_P2, train_all_img_id_P2 = parse_dataset_scheme(train_set_scheme_P2, com_to_img_id_2,dataset_folder=dataset_folder, exclude_img_id=exclude_img_id.union(train_all_img_id_P1).union(val_all_img_id_P1), split='train',trunc_size=sub_train_size)
    val_community_name_to_img_id_P2, val_all_img_id_P2 = parse_dataset_scheme(val_set_scheme_P2, com_to_img_id_2,dataset_folder=dataset_folder, exclude_img_id=train_all_img_id_P2, split='val_out_of_domain',trunc_size=val_sub_size)

    print(train_community_name_to_img_id_P2.keys())
    os.makedirs(os.path.dirname(dataset_folder + '/' + 'imageID_to_group.pkl'), exist_ok=True)
    
    with open(dataset_folder + '/' + 'imageID_to_group.pkl', 'wb') as handle:
        imageID_to_group = dict()
        group_to_imageID = train_community_name_to_img_id_P2.copy()
        group_to_imageID.update(test_community_name_to_img_id)
        group_to_imageID.update(val_community_name_to_img_id_P2)
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
    
    train_set_scheme_all = {
        'dog':{
        'dog(P1)':set([ f'dog({x})' for x in animals_clusters[0]]),
        'dog(P2)':set([ f'dog({x})' for x in animals_clusters[1]])
        },
    'cat':{
        'cat(P1)':set([ f'cat({x})'  for x in animals_clusters[0]]),
        'cat(P2)':set([ f'cat({x})'  for x in animals_clusters[1]])
            }
    }   
    
    com_to_img_id_all = com_to_img_id_1.copy()
    com_to_img_id_all.update(com_to_img_id_2)
    train_community_name_to_img_id = {}

    dataset_folder = CUSTOM_SPLIT_DATASET_FOLDER+'/irm'
    # train_community_name_to_img_id, train_all_img_id = parse_dataset_scheme(train_set_scheme_all,com_to_img_id_all,dataset_folder=dataset_folder, exclude_img_id=test_all_img_id.union(trainsg_dupes), split='train',trunc_size=100)

    os.makedirs(os.path.dirname(dataset_folder + '/' + 'imageID_to_group.pkl'), exist_ok=True)
    train_community_name_to_img_id = {
        'dog(P1)':train_community_name_to_img_id_P1['dog(P11)'].union(train_community_name_to_img_id_P1['dog(P12)']),
        'dog(P2)':train_community_name_to_img_id_P2['dog(P21)'].union(train_community_name_to_img_id_P2['dog(P22)']),
        'cat(P1)':train_community_name_to_img_id_P1['cat(P11)'].union(train_community_name_to_img_id_P1['cat(P12)']),
        'cat(P2)':train_community_name_to_img_id_P2['cat(P21)'].union(train_community_name_to_img_id_P2['cat(P22)']),
    }

    val_community_name_to_img_id = {
        'dog(P1)_val':val_community_name_to_img_id_P1['dog(P11)_val'].union(val_community_name_to_img_id_P1['dog(P12)_val']),
        'dog(P2)_val':val_community_name_to_img_id_P2['dog(P21)_val'].union(val_community_name_to_img_id_P2['dog(P22)_val']),
        'cat(P1)_val':val_community_name_to_img_id_P1['cat(P11)_val'].union(val_community_name_to_img_id_P1['cat(P12)_val']),
        'cat(P2)_val':val_community_name_to_img_id_P2['cat(P21)_val'].union(val_community_name_to_img_id_P2['cat(P22)_val']),
    }
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

    # exit(0)

    ##################################

    for sub_str in train_community_name_to_img_id:
        sub_a = sub_str.split('(')[0]
        dst = sub_a+'/'+sub_str
        copy_images(
            dataset_folder, 'train', dst, use_symlink=False,
            img_IDs =  train_community_name_to_img_id[sub_str]
            )
    for sub_str in val_community_name_to_img_id:
        sub_a = sub_str.split('(')[0]
        dst = sub_a+'/'+sub_str
        copy_images(
            dataset_folder, 'val_out_of_domain', dst, use_symlink=False,
            img_IDs =  val_community_name_to_img_id[sub_str]
            )

    
    ##################################
    # **Quantifying the distance between train and test subsets**
    # Please be advised that before making MetaShift public, 
    # we have made further efforts to reduce the label errors propagated from Visual Genome. 
    # Therefore, we expect a slight change in the exact experiment numbers.  
    ##################################
    
    print('========== Quantifying the distance between train and test subsets ==========')
    test_community_name_to_img_id, _ = parse_dataset_scheme(test_set_scheme, node_name_to_img_id, exclude_img_id=trainsg_dupes, split='test', copy=False)
    # train_community_name_to_img_id, _ = parse_dataset_scheme(train_set_scheme_all, node_name_to_img_id, exclude_img_id=trainsg_dupes, split='train', copy=False)
    additional_test_community_name_to_img_id, _ = parse_dataset_scheme(additional_test_set_scheme, node_name_to_img_id, exclude_img_id=trainsg_dupes, split='test', copy=False)
    
    # exit(0)
    community_name_to_img_id = test_community_name_to_img_id.copy()
    community_name_to_img_id.update(train_community_name_to_img_id)
    # community_name_to_img_id.update(val_community_name_to_img_id)
    print("community_name_to_img_id.keys()", community_name_to_img_id.keys())
    community_name_to_img_id.update(additional_test_community_name_to_img_id)
    # print(community_name_to_img_id.keys())
    dog_community_name_list = sorted(train_set_scheme_all['dog']) + sorted(test_set_scheme['dog']) + sorted(additional_test_set_scheme['dog'])
    # dog_community_name_list = sorted(train_set_scheme['dog']) + sorted(test_set_scheme['dog'])
    N_sets = len(community_name_to_img_id.keys())
    Adjacency_matrix = np.ones((N_sets, N_sets))
    for i, ii in enumerate(community_name_to_img_id.keys()):
        # print("ii: ",ii)
        
        for j,jj in enumerate(community_name_to_img_id.keys()):
            set_A = community_name_to_img_id[ii] - trainsg_dupes
            set_B = community_name_to_img_id[jj] - trainsg_dupes
            overlap_set = set_A.intersection(set_B)
            if ii == 'dog(shelf)':
                print('jj:',jj)
                print("overlap set length: ", len(overlap_set))
                print("length ii: ", len(set_A), " length jj: ",len(set_B))
                print('**')
            if len(overlap_set) == 0:
                edge_weight = 0
            else: 
                edge_weight = len(overlap_set) / min( len(set_A), len(set_B) )
            Adjacency_matrix[i,j] = Adjacency_matrix[j,i] = edge_weight
            
    # G, pars_sc, Adjacency_matrix = build_subset_graph(dog_community_name_list, community_name_to_img_id, trainsg_dupes=set(), subject_str=None,seed=0)
    # G_dog_all, dog_pars_sc, Adjacency_matrix = build_subset_graph(dog_community_name_list, community_name_to_img_id, trainsg_dupes=set(), subject_str=None,seed=0)
    labels = []
    for i, x in enumerate(community_name_to_img_id.keys()):
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
    for sub in train_set_scheme_all['dog']:
        # print('subset name: ', sub)
        subs.append(sub)
        dists.append(np.linalg.norm(spectral_pos[sub.replace('(','\n(')] - spectral_pos['dog\n(shelf)']))
        # distance_B = np.linalg.norm(spectral_pos[sub.replace('(','\n(')] - spectral_pos['dog\n(shelf)'])
        
    
    print('Distance from {}+{} to {}: {}'.format(
            subs[0], subs[1], 'dog(shelf)', 
            0.5 * (dists[0] + dists[1])
            ))
    print(f"Distance from {subs[0]} is {dists[0]}\nDistance from {subs[1]} is {dists[1]} ")
        
    # exit(0)\

    # communities = set(animals_clusters[0]).copy()
    # communities.update(set(animals_clusters[1]))
    # communities.update(test_community_name_to_img_id.keys())
    # com_to_img_id_all.update(test_community_name_to_img_id)

    # print("communitites", communities)

    # N_sets = len(communities)
    # Adjacency_matrix = np.ones((N_sets, N_sets))
    # for i, ii in enumerate(communities):
    #     print("ii: ",ii)
    #     setA = set()
    #     if ('cat' in ii) or ('dog' in ii):
    #         setA.update(com_to_img_id_all[ii])
            
    #     else:
    #         set_A.update(com_to_img_id_all[f'dog({ii})'])
    #         set_A.update(com_to_img_id_all[f'cat({ii})'])
        
    #     for j,jj in enumerate(communities):
    #         setB = set()
    #         if ('cat' in jj) or ('dog' in jj):
    #             setB.update(com_to_img_id_all[jj])
            
    #         else:
    #             set_B.update(com_to_img_id_all[f'dog({jj})'])
    #             set_B.update(com_to_img_id_all[f'cat({jj})'])
    #         overlap_set = set_A.intersection(set_B)
    #         if len(overlap_set) == 0:
    #             edge_weight = 0
    #         else: 
    #             edge_weight = len(overlap_set) / min( len(set_A), len(set_B) )
    #         Adjacency_matrix[i,j] = Adjacency_matrix[j,i] = edge_weight
            
    # labels = []
    # for i, x in enumerate(communities):
    #     # add a \n
    #     labels.append(x.replace('(', '\n('))
    # A_pd = pd.DataFrame(np.matrix(Adjacency_matrix), index=labels, columns=labels)
    # G_dog_all = nx.from_pandas_adjacency(A_pd)
    # print("=== similarity measure ===")
    # for com in [animals_clusters[0], animals_clusters[1]]:
    #     # all_comms = set(new_sub_comms)-set(com)
    #     cut_comm = nx.cut_size(G_dog_all,'dog\n(shelf)',com,weight='weight')
    #     print(f'{com}: {cut_comm}')
    #     # all_cuts.append(cut_comm)
    return

if __name__ == '__main__':
    generate_splitted_metadaset()