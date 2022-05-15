## Description:
1. `dataset\create_subpopulationshift_dataset.py`: a script for creating a partitioned dataset, outputs 3 subfolders `[p1, p2, irm]`
    
    Arguments:
    ```
    --seed:                   int 
    --minority_percentage:    int [truncates the minority group to a percentage p of the majority group]
    --overlap_len:            int [number of duplicated communities in each leaf partition]
    ```
2. `main_experiment.py`: a script that runs the vision model
    Arguments:
    ```
    --experiment:       str ['irm' or 'pirm']
    ```
  
3. `run_main.py`: script the runs both scripts, of creating the dataset and applying the experimnets
  
    Arguments:
    ```
    --algorithm:  str IRM
    --details:    str [experiment ID/name of one's choice]
    --pyfile:     str [main_experiment.py]
    --data:       str [data folder for saving the partitioned dataset]
    --dataset_script: str [two options: 'dataset/create_domain_generalization_dataset.py' OR 'dataset/create_subpopulationshift_dataset'] with its arguments
    --anneals_p1:   int [special IRM penalty annealing epoch for pirm first model]
    --anneals_p2:   int [special IRM penalty annealing epoch for pirm second model]
    --anneals_irm:  int [special IRM penalty annealing epoch for irm model]
    --data_exists:  flag [if data folder exists]
  ```
