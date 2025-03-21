########
#
# unsw-goannas-prepare-yolo-training-set.py
#
# Given the COCO-formatted training set, prepare the final YOLO training data:
#
# * Convert the whole dataset (with train/val still merged) to YOLO format
#
# * Split locations into train/val
#
# * Preview the train/val files to make sure everything looks OK
#
# * Prepare symlinks for the final yolo dataset
#
# * Preview
#
########

#%% Imports and constants

import os
import json

from data_management import coco_to_yolo

input_folder = os.path.expanduser('~/data/unsw-alting/labelme-folders')
output_folder = os.path.expanduser('~/data/unsw-alting/yolo-training-folder')
cct_file = os.path.expanduser('~/data/unsw-alting/labeled-images-cct.json')

assert os.path.isdir(input_folder)
assert os.path.isfile(cct_file)
os.makedirs(output_folder,exist_ok=True)

common_categories = ('goanna','dingo','quoll','fox','possum')


#%% Convert the un-split data to YOLO

coco_to_yolo_info = coco_to_yolo.coco_to_yolo(input_folder,
                         output_folder=input_folder,
                         input_file=cct_file,
                         source_format='coco_camera_traps',
                         overwrite_images=False,
                         create_image_and_label_folders=False,
                         class_file_name='classes.txt',
                         allow_empty_annotations=False,
                         clip_boxes=True,
                         image_id_to_output_image_json_file=None,
                         images_to_exclude=None,
                         path_replacement_char='#',
                         category_names_to_exclude=None,
                         category_names_to_include=common_categories,
                         write_output=True,
                         flatten_paths=False)


#%% Prepare for train/val splitting

from collections import defaultdict

# Compute category counts for each location
with open(cct_file,'r') as f:
    cct_data = json.load(f)

category_id_to_name = {}
for c in cct_data['categories']:
    category_id_to_name[c['id']] = c['name']
    
image_id_to_category_names = defaultdict(set)

# ann = cct_data['annotations'][0]
for ann in cct_data['annotations']:
    image_id = ann['image_id']
    category_name = category_id_to_name[ann['category_id']]
    image_id_to_category_names[image_id].add(category_name)

location_to_category_counts = {}

# im = cct_data['images'][0]
for im in cct_data['images']:
    location_id = im['location']
    if location_id not in location_to_category_counts:
        location_to_category_counts[location_id] = defaultdict(int)
    category_names_this_image = image_id_to_category_names[im['id']]
    for category_name in category_names_this_image:
        location_to_category_counts[location_id][category_name] += 1        

category_to_max_allowable_error = {}
category_to_error_weight = {}
for s in common_categories:
    category_to_max_allowable_error[s] = 0.02
    category_to_error_weight[s] = 2.0
    
category_to_error_weight = {'goanna':5}
default_max_allowable_error = None


#%% Split locations into train/val

from md_utils.split_locations_into_train_val import split_locations_into_train_val

val_locations,category_to_val_fraction = split_locations_into_train_val(
    location_to_category_counts=location_to_category_counts,
    n_random_seeds=10000,
    target_val_fraction=0.15,
    category_to_max_allowable_error=category_to_max_allowable_error,                                   
    category_to_error_weight=category_to_error_weight,
    default_max_allowable_error=default_max_allowable_error)


#%% Create YOLO train/val datasets in symlink folders

# ...excluding whatever images we don't want to use for training.  For now, I'm only
# going to train on images in the common categories.

from md_utils.path_utils import safe_create_link
from tqdm import tqdm

train_folder = os.path.join(output_folder,'train')
val_folder = os.path.join(output_folder,'val')
os.makedirs(train_folder,exist_ok=True)
os.makedirs(val_folder,exist_ok=True)

# im = cct_data['images'][0]
for i_image,im in tqdm(enumerate(cct_data['images']),
                       total=len(cct_data['images'])):
    
    # Only create links for common images
    categories_this_image = image_id_to_category_names[im['id']]    
    includes_common_category = False
    for category in categories_this_image:
        if category in common_categories:
            includes_common_category = True
            break
    if not includes_common_category:
        continue
    
    location_id = im['location']
    if location_id in val_locations:
        target_folder = val_folder
    else:
        target_folder = train_folder
    
    source_image_file_relative = im['file_name']
    source_image_file_abs = os.path.join(input_folder,source_image_file_relative)
    source_annotation_file_relative = os.path.splitext(source_image_file_relative)[0] + '.txt'
    source_annotation_file_abs = os.path.join(input_folder,source_annotation_file_relative)
    assert os.path.isfile(source_image_file_abs) and os.path.isfile(source_annotation_file_abs)
    
    target_image_file_relative = source_image_file_relative.replace('/','#')
    target_annotation_file_relative = source_annotation_file_relative.replace('/','#')
    target_image_file_abs = os.path.join(target_folder,target_image_file_relative)
    target_annotation_file_abs = os.path.join(target_folder,target_annotation_file_relative)
        
    safe_create_link(source_image_file_abs,target_image_file_abs)
    safe_create_link(source_annotation_file_abs,target_annotation_file_abs)

# ...for each image

# Copy the class file to each symlink folder    
class_file_abs = os.path.join(input_folder,'classes.txt')
assert os.path.isfile(class_file_abs)

import shutil
shutil.copyfile(class_file_abs,os.path.join(train_folder,'object.data'))
shutil.copyfile(class_file_abs,os.path.join(val_folder,'object.data'))


#%% Count the total number of files we just wrote

n_source_images = len(cct_data['images'])

train_files = os.listdir(train_folder)
train_annotations = [fn for fn in train_files if fn.endswith('.txt')]

val_files = os.listdir(val_folder)
val_annotations = [fn for fn in val_files if fn.endswith('.txt')]

print('{} val images, {} train images, {} total output images, {} input images'.format(
    len(val_annotations),len(train_annotations),
    len(val_annotations)+len(train_annotations),
    n_source_images))

                
#%% Generate the YOLOv5 dataset.yaml file

yolo_dataset_file = os.path.join(output_folder,'dataset.yml')
coco_to_yolo.write_yolo_dataset_file(yolo_dataset_file,
                                     dataset_base_dir=output_folder,
                                     class_list=class_file_abs,
                                     train_folder_relative='train',
                                     val_folder_relative='val',
                                     test_folder_relative=None)

# import clipboard; clipboard.copy(yolo_dataset_file)


#%% Preview in BoundingBoxEditor...

# ...to make sure things look basically sound

import shutil
class_list_file = os.path.join(input_folder,'classes.txt')
assert os.path.isfile(class_list_file)
# import clipboard; clipboard.copy(class_list_file)
bbe_class_list_file_val = os.path.join(val_folder,'object.data')
shutil.copyfile(class_list_file,bbe_class_list_file_val)
bbe_class_list_file_train = os.path.join(val_folder,'object.data')
shutil.copyfile(class_list_file,bbe_class_list_file_train)
