########
#
# unsw-goannas-explor-data.py
#
# Exploratory analyses performed when I was rehydrating this project in 2025.
#
########


#%% Imports and constants (Windows exploration)

import os

base_folder = r'e:\data\unsw-alting\unsw-alting-yolo-1600'.replace('\\','/')
train_folder = os.path.join(base_folder,'train').replace('\\','/')
val_folder = os.path.join(base_folder,'val').replace('\\','/')

assert os.path.isdir(base_folder)
assert os.path.isdir(train_folder)
assert os.path.isdir(val_folder)


#%% List files

from megadetector.utils.path_utils import recursive_file_list
from megadetector.utils.path_utils import is_image_file

all_files_relative = recursive_file_list(base_folder,return_relative_paths=True)
all_txt_files = [fn for fn in all_files_relative if fn.endswith('.txt')]
all_image_files = [fn for fn in all_files_relative if is_image_file(fn)]
all_train_files = [fn for fn in all_files_relative if 'train/' in fn]
all_val_files = [fn for fn in all_files_relative if 'val/' in fn]

print('Enumerated {} files'.format(len(all_files_relative)))
print('Enumerated {} label files'.format(len(all_txt_files)))
print('Enumerated {} image files'.format(len(all_image_files)))
print('Enumerated {} train files'.format(len(all_train_files)))
print('Enumerated {} val files'.format(len(all_val_files)))

"""
Enumerated 263611 files
Enumerated 131802 label files
Enumerated 131802 image files
Enumerated 227187 train files
Enumerated 36419 val files
"""


#%% Explore .json files

from megadetector.utils.path_utils import insert_before_extension

json_base = 'unsw-original-categories.json'
json_md_classes = \
    insert_before_extension(json_base,'md_classes')
json_md_classes_with_blank_flags = \
    insert_before_extension(json_md_classes,'with_blank_flags')
json_md_classes_with_blank_flags_with_location = \
    insert_before_extension(json_md_classes_with_blank_flags,'with_location')
    
all_json_files = [
    json_base,
    json_md_classes,
    json_md_classes_with_blank_flags,
    json_md_classes_with_blank_flags_with_location
]

for fn_relative in all_json_files:
    assert os.path.isfile(os.path.join(base_folder,fn_relative))


#%% Explore raw data file

import json

json_base_abs = os.path.join(base_folder,json_base)

with open(json_base_abs,'r') as f:
    d = json.load(f)
    

#%% Validate filenames

all_files_relative_set = set(all_files_relative)

for im in d['images']:
    fn_relative = im['file_name'].replace('\\','/')
    assert fn_relative in all_files_relative_set
    
assert len(d['images']) == len(all_files_relative)


#%% Count categories

from collections import defaultdict

image_id_to_n_annotations = defaultdict(int)
category_id_to_image_ids = defaultdict(set)

for ann in d['annotations']:
    image_id_to_n_annotations[ann['image_id']] += 1
    category_id_to_image_ids[ann['category_id']].add(ann['image_id'])
    
for im in d['images']:
    assert im['id'] in image_id_to_n_annotations and image_id_to_n_annotations[im['id']] > 0

category_id_to_name = {c['id']:c['name'] for c in d['categories']}

for category_id in category_id_to_name:
    category_name = category_id_to_name[category_id]
    category_count = len(category_id_to_image_ids[category_id])
    print('{}: {}'.format(category_name,category_count))
    
"""
dingo: 24540
fox: 4421
goanna: 84361
possum: 2074
quoll: 16406
empty: 0
"""

