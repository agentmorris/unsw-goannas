########
#
# unsw-goannas-scrap.py
#
# Exploratory analyses performed around the time of the original model
# training (2023).
# 
########

#%% Imports and constants

import os

base_folder = os.path.expanduser('~/data/unsw-alting')
yolo_folder = os.path.join(base_folder,'yolo-training-folder')
labelme_folder = os.path.join(base_folder,'labelme-folders')

assert os.path.isdir(yolo_folder)
assert os.path.isdir(labelme_folder)


#%% Find images in the YOLO and labelme folders

from md_utils.path_utils import find_images
yolo_images = find_images(yolo_folder,return_relative_paths=True,recursive=True)
labelme_images = find_images(labelme_folder,return_relative_paths=True,recursive=True)

print('Found {} images in yolo folder'.format(len(yolo_images)))
print('Found {} images in labelme folder'.format(len(labelme_images)))


#%% See what's in the labelme folder, but not the yolo folder

# YOLO filenames look like:
# 
# goanna#BrendanAltingMLDP2023Images#Q25#Q25__2023-01-15__12-41-27(10).JPG
#
# labelme filenames look like:
#
# BrendanAltingMLDP2023Images#Q25#Q25__2023-01-15__12-41-27(10).JPG    

yolo_base_filenames = set()
for fn in yolo_images:
    bn = fn.split('/')[-1]
    assert '#' in bn
    bn = '#'.join(bn.split('#')[1:])
    yolo_base_filenames.add(bn)
    
images_not_in_yolo_folder = []    

# fn = labelme_images[0]
for fn in labelme_images:
    bn = fn.split('/')[-1]
    if bn not in yolo_base_filenames:
        images_not_in_yolo_folder.append(fn)        
        
print('Found {} files in the labelme folder that are not in the YOLO folder'.format(
    len(images_not_in_yolo_folder)))


#%% Look for unused labels

from collections import defaultdict
label_to_unused_count = defaultdict(int)
for fn in images_not_in_yolo_folder:
    label = fn.split('/')[0]
    label_to_unused_count[label] = label_to_unused_count[label] + 1
    
print('Unused labels:\n')

from md_utils.ct_utils import sort_dictionary_by_value
label_to_unused_count = sort_dictionary_by_value(label_to_unused_count,reverse=True)

for s in label_to_unused_count:
    print('{}: {}'.format(s,label_to_unused_count[s]))
