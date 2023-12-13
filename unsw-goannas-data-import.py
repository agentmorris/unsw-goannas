#########
#
# unsw-goannas-data-import.py
#
# Parse the source .csv files from the UNSW Goannas dataset into a COCO Camera Traps .json file.
#
#########

#%% Constants and imports

import os
import json
import pandas as pd
import numpy as np

from tqdm import tqdm
from collections import defaultdict

metadata_folder = os.path.expanduser('~/data/unsw-alting')
image_folder = '/datadrive/home/sftp/unsw-alting_/data'

image_list_cache = os.path.join(metadata_folder,'image_list.json')

classified_fn = os.path.join(metadata_folder,'MLDPImagesClassified.csv')
timelapse_export_fn = os.path.join(metadata_folder,'TimelapseTemplateSummer2023.csv')

output_coco_file = os.path.join(metadata_folder,'unsw_goannas.json')

labeling_image_width = 1600
labeling_folder_base = os.path.join(metadata_folder,'labelme-folders')

flag_names = ('animal','empty','person','vehicle')

assert all([os.path.isfile(fn) for fn in (classified_fn,timelapse_export_fn)])
assert os.path.isdir(image_folder)


#%% Enumerate files

from md_utils.path_utils import find_images

if os.path.isfile(image_list_cache):
    with open(image_list_cache,'r') as f:
        all_images = json.load(f)
    print('Loaded enumeration of {} images'.format(len(all_images)))
else:
    all_images = find_images(image_folder,recursive=True,return_relative_paths=True)
    with open(image_list_cache,'w') as f:
        json.dump(all_images,f,indent=1)
    print('Enumerated {} images in {}'.format(len(all_images),image_folder))

all_images_set = set(all_images)

# 1060266 images on disk


#%% Read metadata

classified_df = pd.read_csv(classified_fn)
print('Read {} rows from {}'.format(len(classified_df),classified_fn))

timelapse_df = pd.read_csv(timelapse_export_fn)
print('Read {} rows from {}'.format(len(timelapse_df),timelapse_export_fn))

all_df_info = {
    classified_fn:classified_df,
    timelapse_export_fn:timelapse_df
    }

# 1060266 rows in classified_df
#
# 2090085 rows in timelapse_df
#
# (2*1060266)-2090085 = 30447


#%% Print species counts for both data files

# Nothing from this cell is used later

if False:
    
    for fn in all_df_info.keys():
        
        df = all_df_info[fn]
        
        print('Analyzing {}'.format(fn))
        
        species_to_count = defaultdict(int)
        
        n_person = 0
        n_vehicle = 0
        n_ignored_animal = 0
        n_unannotated = 0
        
        folders_with_goannas = set()
        all_folders = set()
        
        tqdm.pandas()
        
        for i_row,row in tqdm(df.iterrows(),total=len(df)):
            
            all_folders.add(row['RelativePath'])
            
            if isinstance(row['Species'],float):        
                assert np.isnan(row['Species']) 
                if row['person']:
                    n_person += 1
                if row['vehicle']:
                    n_vehicle += 1
                if row['animal']:
                    n_ignored_animal += 1
                if not (row['person'] or row['vehicle'] or row['animal']):
                    n_unannotated += 1
                continue
            
            species = row['Species']    
            assert isinstance(species,str)
            species_to_count[species] = species_to_count[species] + 1
            
            if species == 'Goanna':
                folders_with_goannas.add(row['RelativePath'])
        
        print('')
        print('person: {}'.format(n_person))
        print('vehicle: {}'.format(n_vehicle))
        print('ignored animal: {}'.format(n_ignored_animal))
        print('unannotated: {}'.format(n_unannotated))
        print('')
        
        for k in species_to_count.keys():
            print('{}: {}'.format(k,species_to_count[k]))
            
        print('\n{} of {} folders contain goannas'.format(
            len(folders_with_goannas),len(all_folders)))

## MLDP

"""
person: 1723
vehicle: 139
ignored animal: 10516
unannotated: 917938

Dingo: 24555
Goanna: 82741
Fox: 4429
Quoll: 16382
Cat: 65
Possum: 2074
Dragon sp.: 194
Tawny: 32
Land Mullet: 473
Red-Bellied Black Snake: 20
BirdOfPrey: 109
Brown Snake: 20
Blue-Tounged lizard: 10

63 of 124 folders contain goannas
"""

# Timelapse

"""
Dingo: 24555
Goanna: 84371
Fox: 4429
Quoll: 16416
Cat: 65
Possum: 2074
Dragon sp.: 194
Tawny: 32
Land Mullet: 497
Red-Bellied Black Snake: 20
BirdOfPrey: 109
Brown Snake: 20
Blue-Tounged lizard: 19

63 of 206 folders contain goannas
"""


#%% Columns

# In MLDPImagesClassified.csv

"RootFolder,File,RelativePath,DateTime,DeleteFlag,animal,empty,person,vehicle,Species"

# In the Timelapse template
"RootFolder,File,RelativePath,DateTime,DeleteFlag,animal,empty,person,vehicle,Species"
"Unidentifiable,Dark,No_Adults_Seq,No_Juv_Seq"
"JBU2101,...,QU2205"

 
#%% Confirm that the images in MLDPImagesClassified.csv line up with the images I have

# ...and pull species information from MLDPImagesClassified.csv

# Takes ~1 min

from dateutil import parser

assert len(classified_df) == len(all_images)

top_level_folders_mldp = set()

fn_relative_to_species_mldp = {}
fn_relative_to_datetime_mldp = {}
fn_relative_to_flags_mldp = {}

# i_row = 0; row = classified_df.iloc[i_row]
for i_row,row in tqdm(classified_df.iterrows(),total=len(classified_df)):
    fn_relative = (row['RelativePath'] + '/' + row['File']).replace('\\','/')
    top_level_folders_mldp.add(fn_relative.split('/')[0])
    fn_abs = os.path.join(image_folder,fn_relative)
    assert os.path.isfile(fn_abs)
    assert fn_relative not in fn_relative_to_species_mldp
    fn_relative_to_species_mldp[fn_relative] = row['Species']
    
    dt_str = row['DateTime'].strip()
    dt = parser.parse(dt_str)
    
    assert dt.year >= 2018 and dt.year <= 2023
    fn_relative_to_datetime_mldp[fn_relative] = dt
    
    animal_flag = row['animal']
    assert isinstance(animal_flag,bool)
    flags = {}
    for flag_name in flag_names:
        flag_value = row[flag_name]
        assert isinstance(flag_value,bool)
        flags[flag_name] = flag_value
    fn_relative_to_flags_mldp[fn_relative] = flags
    
# {'BrendanAltingMLDP2023Images', 'PSML2023-06'}


#%% Check for redundancy between the top-level folders

# I.e., make sure that there are no images that exist within both of the top-level folders:
#
# {'BrendanAltingMLDP2023Images', 'PSML2023-06'}

fn_relative_without_top_level_mldp_to_species = {}
fn_relative_without_top_level_mldp_to_fn_relative = {}

# fn_relative = next(iter(fn_relative_to_species_mldp.keys()))
for fn_relative in tqdm(fn_relative_to_species_mldp):
    
    fn_relative_without_top_level = '/'.join(fn_relative.split('/')[1:])
    
    assert fn_relative_without_top_level not in fn_relative_without_top_level_mldp_to_species
    assert fn_relative_without_top_level not in fn_relative_without_top_level_mldp_to_fn_relative
    
    fn_relative_without_top_level_mldp_to_species[fn_relative_without_top_level] = \
        fn_relative_to_species_mldp[fn_relative]
        
    fn_relative_without_top_level_mldp_to_fn_relative[fn_relative_without_top_level] = \
        fn_relative    


#%% Map species classifications from the Timelapse .csv

# Takes ~1 min

top_level_folders_timelapse = set()
fn_relative_to_species_timelapse = {}
fn_relative_to_flags_timelapse = {}

# i_row = 0; row = timelapse_df.iloc[i_row]
for i_row,row in tqdm(timelapse_df.iterrows(),total=len(timelapse_df)):
    
    fn_relative = (row['RelativePath'] + '/' + row['File']).replace('\\','/')
    top_level_folders_timelapse.add(fn_relative.split('/')[0])
    assert fn_relative not in fn_relative_to_species_timelapse
    fn_relative_to_species_timelapse[fn_relative] = row['Species']
    
    flags = {}
    for flag_name in flag_names:
        flag_value = row[flag_name]
        assert isinstance(flag_value,bool)
        flags[flag_name] = flag_value
    fn_relative_to_flags_timelapse[fn_relative] = flags


#%% Make sure images in the MLDP .csv are also represented in the Timelapse .csv

# ...either in their original form or without their top-level folder

for fn_relative in tqdm(fn_relative_to_species_mldp):
    fn_relative_without_top_level = '/'.join(fn_relative.split('/')[1:])
    assert (fn_relative in fn_relative_to_species_timelapse) or \
        (fn_relative_without_top_level in fn_relative_to_species_timelapse)
                
    
#%% Confirm that the filenames in the Timelapse .csv that *look* like the ones in the MLDP .csv actually match

filenames_that_match_mldp = set()
filenames_that_match_mldp_after_top_level_correction = set()

# fn_relative = next(iter(fn_relative_to_species_timelapse.keys()))
for fn_relative in tqdm(fn_relative_to_species_timelapse):
    
    top_level_folder = fn_relative.split('/')[0]
    
    # If this top level folder exists in both .csv files, make sure the 
    # relative filenames match exactly
    if top_level_folder in top_level_folders_mldp:
        assert fn_relative in fn_relative_to_species_mldp
        filenames_that_match_mldp.add(fn_relative)
    # Otherwise make sure this appears in the MLDP file after removing the top-level
    # folder
    elif fn_relative in fn_relative_without_top_level_mldp_to_species:
        filenames_that_match_mldp_after_top_level_correction.add(fn_relative)
    else:
        raise ValueError('Could not match {} to MLDP file'.format(fn_relative))

assert filenames_that_match_mldp == set(fn_relative_to_species_mldp.keys())

print('\nOf {} files in the Timelapse .csv:'.format(
    len(fn_relative_to_species_timelapse)))
print('{} match the MLDP .csv'.format(
    len(filenames_that_match_mldp)))
print('{} match the MLDP .csv after correction'.format(
    len(filenames_that_match_mldp_after_top_level_correction)))


#%% Find images that were corrected only in one file or the other

def species_equal(a,b):
    if isinstance(a,float) and isinstance(b,float):
        return (np.isnan(a) and np.isnan(b))
    elif isinstance(a,str) and isinstance(b,str):
        return (a == b)
    else:
        return False

# Map relative filenames on disk to species that are available only in the Timelapse
# .csv file... this is important since we'll be walking through the MLDP file when 
# we generate our output file, and we'll use this as a backup.
fn_relative_to_species_only_in_timelapse = {}

# Map relative filenames on disk to species that are available only in the Timelapse
# .csv file.  This is not used later, it's just a consistency check.
fn_relative_to_species_only_in_mldp = {}

# It appears in practice that the annotations appearing only in the timelapse file are 
# always annotations that don't require top-level folder correction, and the annotations 
# appearing only in the MLDP file always *do* require top-level folder correction.

# fn_relative = next(iter(filenames_that_match_mldp))
for fn_relative in tqdm(filenames_that_match_mldp):
    species_mldp = fn_relative_to_species_mldp[fn_relative]
    species_timelapse = fn_relative_to_species_timelapse[fn_relative]
    if not species_equal(species_mldp,species_timelapse):
        assert np.isnan(species_mldp)
        assert isinstance(species_timelapse,str)
        fn_relative_to_species_only_in_timelapse[fn_relative] = species_timelapse

# fn_relative_without_top_level = next(iter(filenames_that_match_mldp_after_top_level_correction))
for fn_relative_without_top_level in tqdm(filenames_that_match_mldp_after_top_level_correction):
    fn_relative = fn_relative_without_top_level_mldp_to_fn_relative[fn_relative_without_top_level]
    species_mldp = fn_relative_to_species_mldp[fn_relative]
    species_timelapse = fn_relative_to_species_timelapse[fn_relative_without_top_level]
    if not species_equal(species_mldp,species_timelapse):
        assert np.isnan(species_timelapse)
        assert isinstance(species_mldp,str)
        fn_relative_to_species_only_in_mldp[fn_relative] = species_mldp
    
print('\n{} annotations are only in the Timelapse file'.format(len(fn_relative_to_species_only_in_timelapse)))
print('{} annotations are only in the MLDP file'.format(len(fn_relative_to_species_only_in_mldp)))

for fn_relative in fn_relative_to_species_only_in_timelapse:
    assert fn_relative in all_images_set
    assert fn_relative not in fn_relative_to_species_only_in_mldp
    
for fn_relative in fn_relative_to_species_only_in_mldp:
    assert fn_relative in all_images_set
    assert fn_relative not in fn_relative_to_species_only_in_timelapse


#%% Look at one of the corrected files and make sure it's legit

corrected_filenames = sorted(list(fn_relative_to_species_only_in_timelapse.keys()))

for i_file,fn_relative in enumerate(corrected_filenames):
    species = fn_relative_to_species_only_in_timelapse[fn_relative]
    if species != 'Goanna':
        print('{}: {}'.format(i_file,species))
        
        
#%% Inspect one image

i_file = 933  
fn_relative = corrected_filenames[i_file]
fn_abs = os.path.join(image_folder,fn_relative)

print('Annotations say... {}'.format(fn_relative_to_species_only_in_timelapse[fn_relative]))
from md_utils.path_utils import open_file
open_file(fn_abs)


#%% Parse camera locations from filenames

import re

locations = set()
fn_relative_to_location = {}

# i_fn = 0; fn_relative = all_images[i_fn]
for i_fn,fn_relative in tqdm(enumerate(all_images),total=len(all_images)):

    location = None
    pat_ps = 'PS\d+/Cam[AB]'
    m = re.search(pat_ps,fn_relative)
    if m is not None:
        location = m.group()
    else:
        m = re.search('/(Q\d+)/',fn_relative)
        if m is not None:
            location = m.groups(0)[0]
    assert location is not None
    locations.add(location)
    fn_relative_to_location[fn_relative] = location
    
    
#%% Generate a COCO Camera Traps file

from md_visualization.visualization_utils import open_image

images = []
annotations = []

category_name_to_id = {}

debug_max_file = -1

read_image_size = True

# i_fn = 0; fn_relative = all_images[i_fn]
for i_fn,fn_relative in tqdm(enumerate(all_images),total=len(all_images)):
    
    if debug_max_file > 0 and i_fn >= debug_max_file:
        break
    
    im = {}
    im['id'] = fn_relative
    im['file_name'] = fn_relative
    im['location'] = fn_relative_to_location[fn_relative]
    
    dt_string = str(fn_relative_to_datetime_mldp[fn_relative])
    im['datetime'] = dt_string
    
    fn_abs = os.path.join(image_folder,fn_relative)
    if read_image_size:
        try:
            pil_im = open_image(fn_abs)
            im['width'] = pil_im.width
            im['height'] = pil_im.height
        except Exception:
            im['width'] = -1
            im['height'] = -1
    
    species = fn_relative_to_species_mldp[fn_relative]
    if fn_relative in fn_relative_to_species_only_in_timelapse:
        assert isinstance(species,float) and np.isnan(species)
        species = fn_relative_to_species_only_in_timelapse[fn_relative]
        assert isinstance(species,str)
        
    # My original understanding was that if MD thought this image was a 
    # positive, but it's marked empty, it's very likely empty.  I no longer
    # think this is true, it basically just means it wasn't a species of 
    # interest.
    if False:
        if isinstance(species,float):
            assert np.isnan(species)
                    
            if False:
                flags = fn_relative_to_flags_mldp[fn_relative]
                if flags['animal']:
                    species = 'reviewed_empty'
                else:
                    species = 'unannotated'
            species = 'unannotated'
    
    # Then I came to believe that the "empty" checkbox was the one I should be looking at,
    # but now I don't think that's the case either.
    if False:
        flags = fn_relative_to_flags_mldp[fn_relative]
        if isinstance(species,float):
            if flags['empty']:
                species = 'empty'
            else:
                species = 'unannotated'
        else:
            assert not flags['empty']
    
    if isinstance(species,float):
        assert np.isnan(species)
        species = 'unannotated'
        
    species = species.lower().replace(' ','_').replace('.','')
    
    if species not in category_name_to_id:
        category_name_to_id[species] = len(category_name_to_id) + 1
    
    category_id = category_name_to_id[species]
            
    ann = {}
    ann['id'] = fn_relative
    ann['image_id'] = fn_relative
    ann['category_id'] = category_id
    ann['sequence_level_annotation'] = False
    
    annotations.append(ann)
    images.append(im)

# ...for each file

info = {'description':'UNSW Goannas','version':'1.0.0'}
categories = []
for category_name in category_name_to_id:
    categories.append({'id':category_name_to_id[category_name],'name':category_name})
    
d = {}
d['info'] = info
d['images'] = images
d['annotations'] = annotations
d['categories'] = categories

with open(output_coco_file,'w') as f:
    json.dump(d,f,indent=1)
    
    
#%% Integrity check

from data_management.databases.integrity_check_json_db import IntegrityCheckOptions
from data_management.databases.integrity_check_json_db import integrity_check_json_db

options = IntegrityCheckOptions()
options.baseDir = image_folder
bCheckImageSizes = False
bCheckImageExistence = True
bFindUnusedImages = True
bRequireLocation = True

sortedCategories, data, errorInfo = integrity_check_json_db(output_coco_file,options)


#%% Preview

from md_visualization.visualize_db import DbVizOptions
from md_visualization.visualize_db import visualize_db

options = DbVizOptions()
options.num_to_visualize = 100
options.viz_size = (700, -1)
options.classes_to_exclude = ['unannotated']
# options.classes_to_include = ['reviewed_empty']
options.parallelize_rendering = True

html_output_file,_ = visualize_db(output_coco_file, os.path.join(metadata_folder,'coco_preview'),
                                            image_folder, options=options)

from md_utils.path_utils import open_file
open_file(html_output_file)
