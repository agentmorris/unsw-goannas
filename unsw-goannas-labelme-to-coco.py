########
#
# unsw-goannas-labelme-to-coco.py
#
# Convert a folder full of labelme .json files (in which folders indicate species, but 
# each .json file just contains the class label "animal") to a COCO Camera Traps file
# with species labels.
#
########

#%% Imports and constants

import os

input_folder = os.path.expanduser('~/data/unsw-alting/labelme-folders')
cct_file = os.path.expanduser('~/data/unsw-alting/unsw-goannas-training.json')

assert os.path.isdir(input_folder)


#%% Convert labelme to COCO

from data_management.labelme_to_coco import labelme_to_coco

_ = labelme_to_coco(input_folder=input_folder,
                    output_file=cct_file,
                    category_id_to_category_name=None,
                    empty_category_name='empty',
                    empty_category_id=None,
                    info_struct=None,
                    relative_paths_to_include=None,
                    relative_paths_to_exclude=None,
                    use_folders_as_labels=True,
                    recursive=True,
                    no_json_handling='error',
                    validate_image_sizes=True,
                    right_edge_quantization_threshold=0.015)


#%% Read the file back, add locations to each image, and write it back out

from collections import defaultdict
from tqdm import tqdm

import json

with open(cct_file,'r') as f:
    cct_data = json.load(f)

def fn_to_location(fn_relative):
    
    tokens = fn_relative.split('#')
    if len(tokens) == 4:
        assert tokens[1].startswith('P')
        return tokens[1] + '_' + tokens[2]
    else:
        assert len(tokens) == 3
        assert tokens[1].startswith('Q')
        return tokens[1]

location_to_files = defaultdict(list)

im = cct_data['images'][0]
for im in tqdm(cct_data['images']):
    location_id = fn_to_location(im['file_name'])
    im['location'] = location_id
    location_to_files[location_id].append(im['file_name'])
    
location_to_files = {k: v for k, v in sorted(
    location_to_files.items(), key=lambda item: len(item[1]))}    

for location_id in location_to_files:
    print('{}: {}'.format(location_id,len(location_to_files[location_id])))

with open(cct_file,'w') as f:
    json.dump(cct_data,f,indent=1)
    
    
#%% Validate

from data_management.databases import integrity_check_json_db

options = integrity_check_json_db.IntegrityCheckOptions()
    
options.baseDir = input_folder
options.bCheckImageSizes = True
options.bCheckImageExistence = True
options.bFindUnusedImages = True
options.bRequireLocation = True

sortedCategories, _, errorInfo = integrity_check_json_db.integrity_check_json_db(cct_file,options)    


#%% Preview

from md_visualization import visualize_db
from md_utils import path_utils

options = visualize_db.DbVizOptions()
options.parallelize_rendering = True
options.include_filename_links = True
options.show_full_paths = True
options.viz_size = (1280,-1)
options.num_to_visualize = 15000
options.htmlOptions['maxFiguresPerHtmlFile'] = 5000
html_file,_ = visualize_db.visualize_db(cct_file,os.path.expanduser('~/tmp/labelme_to_coco_preview'),
                                        input_folder,options)

path_utils.open_file(html_file)
