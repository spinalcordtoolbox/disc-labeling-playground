"""
Script based on https://github.com/spinalcordtoolbox/disc-labeling-hourglass
"""

import os
import json
import argparse
import subprocess
import numpy as np

from ply.data_management.utils import get_img_path_from_label_path, get_cont_path_from_other_cont

def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='GET or DROP data from git-annex servers. A data config file must be specified.')
    parser.add_argument('--config', required=True, help='Config JSON file where every image used for TRAINING, VALIDATION and TESTING has its path specified ~/<your_path>/config_data.json (Required)')
    parser.add_argument('--drop', action='store_true', help='If specified the paths will be dropped instead of get.')
    parser.add_argument('--clone-repos', action='store_true', help='If specified all missing repositories will be cloned. The data must be following BIDS standards. Neuropoly feature only. (Default=False)')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Read json file and create a dictionary
    with open(args.config, "r") as file:
        config = json.load(file)
    
    # Fetch paths from dictionaries
    paths_to_process = []
    for key in ['TRAINING', 'VALIDATION', 'TESTING']:
        for d in config[key]:
            for rel_path in d.values():
                paths_to_process.append(os.path.join(config['DATASETS_PATH'], rel_path))

    # Extract BIDS repositories path
    repositories_path_list = np.unique([path.split('/derivatives')[0].split('/sub')[0] for path in paths_to_process]) # Splits with /sub and /derivatives to handle both image path and label path
    
    # Clone missing repositories
    for rep_path in repositories_path_list:
        if not os.path.exists(rep_path):
            # Clone missing repositories
            if args.clone_repos:
                subprocess.check_call([
                    'git', 'clone', f'git@data.neuro.polymtl.ca:datasets/{os.path.basename(rep_path)}', rep_path
                ])
            else:
                raise ValueError(f'The git annex repository {rep_path} is missing. Use the flag --clone-repos to clone automatically.')
    
    # GET or DROP paths
    command = 'get' if not args.drop else 'drop'
    for path in paths_to_process:
        rep_path = path.split('/derivatives')[0].split('/sub')[0] # Fetch repository path
        rel_path = path.split(rep_path + '/')[-1] # Fetch relative path
        subprocess.check_call([
                'git', '-C', rep_path, 'annex', command, rel_path
            ])

                



if __name__ == '__main__':
    main()