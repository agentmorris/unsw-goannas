#########
#
# unsw-goannas-labeling.py
#
# Run MD results and species labels through labelme to end up with a set of bounding boxes
# in labelme format, all with category "animal".
#
#########

#%% Constants and imports

import os
import json
import uuid

import clipboard

from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
from datetime import datetime

from md_visualization.visualization_utils import open_image
from md_visualization.visualization_utils import resize_image
from md_utils.path_utils import safe_create_link
from md_utils.path_utils import find_images
from md_utils.ct_utils import split_list_into_fixed_size_chunks
from md_visualization import visualize_db
from md_utils import path_utils
from data_management.databases import integrity_check_json_db
from data_management.labelme_to_coco import labelme_to_coco
from api.batch_processing.postprocessing.md_to_labelme import get_labelme_dict_for_image
from detection.run_detector import DEFAULT_DETECTOR_LABEL_MAP

metadata_folder = os.path.expanduser('~/data/unsw-alting')
image_folder = '/datadrive/home/sftp/unsw-alting_/data'

input_coco_file = os.path.join(metadata_folder,'unsw_goannas.json')

labeling_image_width = 1600
labeling_folder_base = os.path.join(metadata_folder,'labelme-folders')

flag_names = ('animal','empty','person','vehicle')

assert os.path.isdir(image_folder)

max_images_per_batch = 5000
batch_folder_base = os.path.join(metadata_folder,'label_batches')


#%% Load COCO data

with open(input_coco_file,'r') as f:
    coco_data = json.load(f)


#%% Map species to images

image_id_to_image = {im['id']:im for im in coco_data['images']}
category_id_to_name = {c['id']:c['name'] for c in coco_data['categories']}
species_to_relative_paths = defaultdict(list)

species_names_to_annotate = ('goanna','dingo','quoll','fox','possum',
                              'land_mullet','dragon_sp','birdofprey',
                              'cat','tawny','red-bellied_black_snake',
                              'brown_snake','blue-tounged_lizard')

skipped_species = set()

# ann = coco_data['annotations'][0]
for ann in coco_data['annotations']:
    im = image_id_to_image[ann['image_id']]
    fn_relative = im['file_name']
    species = category_id_to_name[ann['category_id']]
    if species not in species_names_to_annotate:
        skipped_species.add(species)
        continue
    species_to_relative_paths[species].append(fn_relative)
    

#%% Copy (and shrink) files to species-specific folders for labeling

overwrite_images = False

# species_name = species_names_to_annotate[0]
for species_name in species_names_to_annotate:
    
    input_relative_paths = species_to_relative_paths[species_name]
    
    print('Resizing {} images of {}'.format(
        len(input_relative_paths),species_name))
    
    species_folder = os.path.join(labeling_folder_base,species_name)
    os.makedirs(species_folder,exist_ok=True)
    
    # input_fn_relative = input_relative_paths[0]
    for input_fn_relative in tqdm(input_relative_paths):
        
        output_fn_base = input_fn_relative.replace('/','#')
        output_fn_abs = os.path.join(species_folder,output_fn_base)
        input_fn_abs = os.path.join(image_folder,input_fn_relative)
        
        if os.path.isfile(output_fn_abs) and (not overwrite_images):
            continue
        
        _ = resize_image(input_fn_abs,target_width=labeling_image_width,
                     target_height=-1,output_file=output_fn_abs)
        # open_file(output_fn_abs)


#%% Load MD results

md_file = os.path.expanduser('~/postprocessing/unsw-alting/unsw-alting-2023-11-02-aug-v5a.0.0/combined_api_outputs/unsw-alting-2023-11-02-aug-v5a.0.0_detections.filtered_rde_0.045_0.850_10_1.010.json')

with open(md_file,'r') as f:
    md_results = json.load(f)
    
# Map filenames to MD results
fn_to_md_results = {im['file']:im for im in md_results['images']}

md_model_str = md_results['info']['detector']


#%% Load results for the intermediate goanna model

goanna_model_results_file = '/home/user/tmp/unsw-alting/inference-results/all-goannas-unsw-goannas-transient-20231112-b8-img1280-e100/all-goannas-unsw-goannas-transient-20231112-b8-img1280-e100.json'

goanna_model_results_file = '/home/user/tmp/unsw-alting/inference-results/all-goannas-unsw-goannas-transient-yolov8-20231114-b-1-img640-e200/all-goannas-unsw-goannas-transient-yolov8-20231114-b-1-img640-e200.json'

goanna_model_results_file = '/home/user/tmp/unsw-alting/inference-results/all-goannas-unsw-goannas-transient-yolov8-20231114195042-b-1-img640-e200/all-goannas-unsw-goannas-transient-yolov8-20231114195042-b-1-img640-e200-aug.json'

goanna_model_results_file = '/home/user/tmp/unsw-alting/inference-results/all-goannas-unsw-goannas-transient-yolov8-2023111615429-b-1-img1280-e200/all-goannas-unsw-goannas-transient-yolov8-2023111615429-b-1-img1280-e200-aug.json'

goanna_model_results_file = '/home/user/tmp/unsw-alting/inference-results/all-goannas-unsw-goannas-transient-yolov8-2023111620736-b-1-img640-e200/all-goannas-unsw-goannas-transient-yolov8-2023111620736-b-1-img640-e200-aug.json'

goanna_model_results_file = '/home/user/tmp/unsw-alting/inference-results/all-goannas-unsw-goannas-transient-yolov8-2023111713648-b-1-img640-e200/all-goannas-unsw-goannas-transient-yolov8-2023111713648-b-1-img640-e200.json'

goanna_model_results_file = '/home/user/tmp/unsw-alting/inference-results/all-goannas-unsw-goannas-transient-yolov8-20231118194424-b-1-img640-e200/all-goannas-unsw-goannas-transient-yolov8-20231118194424-b-1-img640-e200.json'

with open(goanna_model_results_file,'r') as f:
    goanna_model_results = json.load(f)
    
fn_to_goanna_model_results = {im['file']:im for im in goanna_model_results['images']}
    
goanna_model_str = goanna_model_results['info']['detector']

print('Loaded results for {} images'.format(len(goanna_model_results['images'])))


#%% Choose a species for labeling

species_name_to_confidence_threshold = {
    'goanna':[0.25,0.005],
    'dingo':0.2,
    'quoll':0.2,
    'fox':0.2,
    'possum':0.2,
    'land_mullet':0.08,
    'dragon_sp':0.1,
    'birdofprey':0.15,
    'cat':0.2,
    'tawny':0.15,
    'red-bellied_black_snake':0.08,
    'brown_snake':0.08,
    'blue-tounged_lizard':0.08
}

# species_name = 'dingo'
# species_name = 'quoll'
# species_name = 'fox'
# species_name = 'possum'
# species_name = 'land_mullet'
# species_name = 'dragon_sp'
# species_name = 'birdofprey'
# species_name = 'cat'
# species_name = 'tawny'
# species_name = 'red-bellied_black_snake'
# species_name = 'brown_snake'
species_name = 'blue-tounged_lizard'
# species_name = 'goanna'
target_folder = os.path.join(labeling_folder_base,species_name)
assert os.path.isdir(target_folder)

confidence_threshold = species_name_to_confidence_threshold[species_name]
alt_confidence_threshold = 0

if isinstance(confidence_threshold,list) or isinstance(confidence_threshold,tuple):
    alt_confidence_threshold = confidence_threshold[1]
    confidence_threshold = confidence_threshold[0]


#%% Prepare labelme files

destination_image_files = find_images(target_folder,return_relative_paths=True)
source_file_to_destination_image_file = {}

category_id_to_name = md_results['detection_categories']

if species_name != 'goanna':
    info = md_results['info']
else:
    info = goanna_model_results['info']

json_files_written = []
json_files_skipped = []

# 'skip', 'overwrite', 'keepdirty'
overwrite_behavior = 'keepdirty'
max_detections_per_image = None
write_alt_label_files = True

datetime_str = str(datetime.now())

# destination_image_file_relative = destination_image_files[0]
for destination_image_file_relative in tqdm(destination_image_files):

    source_file_relative = destination_image_file_relative.replace('#','/')
    source_file_abs = os.path.join(image_folder,source_file_relative)
    assert os.path.isfile(source_file_abs)
    
    if species_name != 'goanna':
        im = fn_to_md_results[source_file_relative]
        model_str = md_model_str
    else:
        im = fn_to_goanna_model_results[destination_image_file_relative]
        model_str = goanna_model_str
    
    destination_image_file_abs = os.path.join(target_folder,destination_image_file_relative)
    pil_im = open_image(destination_image_file_abs)
    im['width'] = pil_im.width
    im['height'] = pil_im.height
    
    im_copy = deepcopy(im)
    
    if (max_detections_per_image is not None) and (max_detections_per_image >= 1) and \
        (len(im_copy['detections']) > max_detections_per_image):
            im_copy['detections'] = sorted(im_copy['detections'], 
                                           key=lambda d: d['conf'], reverse=True) 
            im_copy['detections'] = im_copy['detections'][0:max_detections_per_image]
            
    labelme_dict = get_labelme_dict_for_image(im_copy,destination_image_file_relative,
                                              category_id_to_name,info,
                                              confidence_threshold=confidence_threshold)
    
    labelme_dict_alt = get_labelme_dict_for_image(im,destination_image_file_relative,
                                              category_id_to_name,info,
                                              confidence_threshold=alt_confidence_threshold)
    
    labelme_dict['created_at'] = datetime_str
    labelme_dict_alt['created_at'] = datetime_str
    
    labelme_dict['model_name'] = model_str
    labelme_dict_alt['model_name'] = model_str
    
    target_metadata_file_relative = os.path.splitext(destination_image_file_relative)[0] + '.json'
    target_metadata_file_relative_alt = os.path.splitext(destination_image_file_relative)[0] + '.alt.json'
    target_metadata_file_abs = os.path.join(target_folder,target_metadata_file_relative)
    target_metadata_file_abs_alt = os.path.join(target_folder,target_metadata_file_relative_alt)
    
    if os.path.isfile(target_metadata_file_abs):
        if (overwrite_behavior == 'skip'):
            json_files_skipped.append(target_metadata_file_abs)
            continue
        elif (overwrite_behavior == 'overwrite'):
            pass
        elif (overwrite_behavior == 'keepdirty'):
            try:
                with open(target_metadata_file_abs,'r') as f:
                    target_metadata = json.load(f)
                    if 'saved_by_labelme' in target_metadata:
                        json_files_skipped.append(target_metadata_file_abs)
                        continue
            except Exception:
                print('Warning: error loading json file {}'.format(target_metadata_file_abs))
                # Write the .json file in this case
                pass
        else:
            raise ValueError('Unrecognized overwrite behavior {}'.format(overwrite_behavior))
    
    json_files_written.append(target_metadata_file_abs)
    with open(target_metadata_file_abs,'w') as f:
        json.dump(labelme_dict,f,indent=1)
    if write_alt_label_files:
        with open(target_metadata_file_abs_alt,'w') as f:
            json.dump(labelme_dict_alt,f,indent=1)

# ...for each image

print('\nWrote .json files for {} images (skipped {})'.format(
    len(json_files_written),len(json_files_skipped)))


#%% Label a species

cmd = 'python labelme {} --labels animal --linewidth 8 --last_updated_file ~/labelme-last-updated.txt'.format(
    target_folder)
print(cmd)
clipboard.copy(cmd)


#%% Resume labeling for a species

cmd = 'python labelme {} --labels animal --linewidth 8 --last_updated_file ~/labelme-last-updated.txt --resume_from_last_update'.format(
    target_folder)
print(cmd)
clipboard.copy(cmd)


#%% Use symlinks to split this species into batches

# ...because labelme gets slow with > ~5000 images

species_batch_folder_base = os.path.join(batch_folder_base,species_name)
os.makedirs(species_batch_folder_base,exist_ok=True)

n_images = len(destination_image_files)

if False:
    L = list(range(0,11))
    split_list_into_fixed_size_chunks(L,4)

chunks = split_list_into_fixed_size_chunks(destination_image_files,max_images_per_batch)

chunk_folders = []

# i_chunk = 0; chunk = chunks[i_chunk]
for i_chunk,chunk in enumerate(chunks):
    
    chunk_folder_abs = os.path.join(species_batch_folder_base,'chunk_{}'.format(
        str(i_chunk).zfill(3)))
    os.makedirs(chunk_folder_abs,exist_ok=True)
    chunk_folders.append(chunk_folder_abs)
    
    # image_file_relative = chunk[0]
    for image_file_relative in tqdm(chunk):
        
        label_file_relative = os.path.splitext(image_file_relative)[0] + '.json'
        label_file_relative_alt = os.path.splitext(image_file_relative)[0] + '.alt.json'
        
        source_image_file_abs = os.path.join(target_folder,image_file_relative)
        source_label_file_abs = os.path.join(target_folder,label_file_relative)
        source_label_file_abs_alt = os.path.join(target_folder,label_file_relative_alt)
        
        assert os.path.isfile(source_image_file_abs) and os.path.isfile(source_label_file_abs)
        
        target_image_file_abs = os.path.join(chunk_folder_abs,image_file_relative)
        target_label_file_abs = os.path.join(chunk_folder_abs,label_file_relative)
        target_label_file_abs_alt = os.path.join(chunk_folder_abs,label_file_relative_alt)
        
        safe_create_link(source_image_file_abs,target_image_file_abs)
        safe_create_link(source_label_file_abs,target_label_file_abs)
        
        if os.path.isfile(source_label_file_abs_alt):
            safe_create_link(source_label_file_abs_alt,target_label_file_abs_alt)

        # s = 'ln -s "{}" "{}"'.format(source_image_file_abs,target_image_file_abs); clipboard.copy(s)
        
    # ...for each image in this chunk
    
# ...for each chunk
  

#%% Label one chunk

i_chunk = 16
species_batch_folder_base = os.path.join(batch_folder_base,species_name)
chunk_folder = os.path.join(species_batch_folder_base,'chunk_{}'.format(
    str(i_chunk).zfill(3)))
cmd = 'python labelme {} --labels animal --linewidth 8 --last_updated_file ~/labelme-last-updated.txt'.format(
    chunk_folder)
print(cmd)
clipboard.copy(cmd)

  
#%% Resume one chunk

i_chunk = None
species_batch_folder_base = os.path.join(batch_folder_base,species_name)
chunk_folder = os.path.join(species_batch_folder_base,'chunk_{}'.format(
    str(i_chunk).zfill(3)))
cmd = 'python labelme {} --labels animal --linewidth 8 --last_updated_file ~/labelme-last-updated.txt --resume_from_last_update'.\
    format(chunk_folder)
print(cmd)
clipboard.copy(cmd)


#%% Validate labelme files (for one species)

target_folder_images = find_images(target_folder,return_relative_paths=True)
target_folder_jsons = [fn for fn in os.listdir(target_folder) if fn.endswith('.json')]

target_folder_jsons_set = set(target_folder_jsons)
target_folder_images_set_no_extension = set([os.path.splitext(fn)[0] for fn in target_folder_images])

# This is a problem, it means we didn't annotate this image
for fn in target_folder_images:
    expected_json = os.path.splitext(fn)[0] + '.json'
    if expected_json not in target_folder_jsons_set:
        print('Could not find .json file for image {}'.format(fn))
        raise Exception('Missing .json')

# This is OK, it happens any time we delete the images after initially saving an annotation
for fn in target_folder_jsons:
    expected_image_no_extension = os.path.splitext(fn)[0]
    if expected_image_no_extension not in target_folder_images_set_no_extension:
        # print('Could not find image file for .json {}'.format(fn))
        pass

# image_fn_relative = target_folder_images[0]
total_shapes = 0

multi_box_files = []
empty_files = []
non_reviewed_files = []
illegal_box_files = []

expected_label = 'animal'
if 'human' in target_folder:
    expected_label = 'person'
    
for image_fn_relative in tqdm(target_folder_images):
    expected_json_relative = os.path.splitext(image_fn_relative)[0] + '.json'
    expected_json_abs = os.path.join(target_folder,expected_json_relative)
    image_fn_abs = os.path.join(target_folder,image_fn_relative)
    with open(expected_json_abs,'r') as f:
        d = json.load(f)
    if 'saved_by_labelme' not in d:
        # 'Image {} was never reviewed'.format(image_fn_relative)
        non_reviewed_files.append(image_fn_relative)
    elif len(d['shapes']) > 1:
        multi_box_files.append(image_fn_relative)
    elif len(d['shapes']) == 0:
        empty_files.append(image_fn_relative)
    for shape in d['shapes']:
        if shape['label'] != expected_label:
            illegal_box_files.add(image_fn_relative)
            print('Illegal label {} in {}'.format(shape['label'],image_fn_abs))
        assert len(shape['points']) == 2
        
print('\n{} files total'.format(len(target_folder_images)))
print('{} files were not reviewed'.format(len(non_reviewed_files)))
print('{} files have multiple boxes'.format(len(multi_box_files)))
print('{} files have no boxes'.format(len(empty_files)))
print('{} files have illegal boxes'.format(len(illegal_box_files)))

# clipboard.copy(os.path.join(target_folder,empty_files[3]))
# clipboard.copy(os.path.join(target_folder,illegal_box_files[0]))


#%% Which chunks were not reviewed?

non_reviewed_files_set = set(non_reviewed_files)

for i_chunk,chunk in enumerate(chunks):
    n_non_reviewed = 0
    for fn in chunk:
        if fn in non_reviewed_files_set:
            n_non_reviewed += 1
    print('Chunk {}: {} non-reviewed images'.format(i_chunk,n_non_reviewed))


#%% Double-check all the empty/illegal files in a symlink folder

tmp_review_folder = os.path.join(batch_folder_base,'tmp_batches',str(uuid.uuid1()))
os.makedirs(tmp_review_folder,exist_ok=True)

files_to_include_relative = illegal_box_files + empty_files + multi_box_files

# image_file_relative = files_to_include_relative[0]
for image_file_relative in tqdm(files_to_include_relative):
    
    label_file_relative = os.path.splitext(image_file_relative)[0] + '.json'
    source_image_file_abs = os.path.join(target_folder,image_file_relative)
    source_label_file_abs = os.path.join(target_folder,label_file_relative)
    assert os.path.isfile(source_image_file_abs) and os.path.isfile(source_label_file_abs)
    
    target_image_file_abs = os.path.join(tmp_review_folder,image_file_relative)
    target_label_file_abs = os.path.join(tmp_review_folder,label_file_relative)
    
    safe_create_link(source_image_file_abs,target_image_file_abs)
    safe_create_link(source_label_file_abs,target_label_file_abs)

cmd = 'python labelme {} --labels animal --linewidth 8 --last_updated_file ~/labelme-last-updated.txt'.format(
    tmp_review_folder)
print(cmd)
clipboard.copy(cmd)


#%% Delete images with no boxes

empty_files_abs = [os.path.join(target_folder,fn) for fn in empty_files]
print('Deleting {} files'.format(len(empty_files_abs)))

for fn_abs in empty_files_abs:
    os.remove(fn_abs)

    
#%% Convert labelme to COCO, preview (for one species)

# image_paths_to_include = multi_box_files
image_paths_to_include = target_folder_images

category_id_to_category_name = DEFAULT_DETECTOR_LABEL_MAP
output_file = output_file = os.path.expanduser('~/tmp/label_validation.json')
output_dict = labelme_to_coco(target_folder,output_file,
                              category_id_to_category_name=category_id_to_category_name,
                              relative_paths_to_include=image_paths_to_include)


##%% Validate

options = integrity_check_json_db.IntegrityCheckOptions()
    
options.baseDir = target_folder
options.bCheckImageSizes = True
options.bCheckImageExistence = True
options.bFindUnusedImages = True
options.bRequireLocation = False

sortedCategories, _, errorInfo = integrity_check_json_db.integrity_check_json_db(output_file,options)    


##%% Preview

options = visualize_db.DbVizOptions()
options.parallelize_rendering = True
options.include_filename_links = True
options.show_full_paths = True
options.viz_size = (1280,-1)
options.num_to_visualize = 5000

html_file,_ = visualize_db.visualize_db(output_file,os.path.expanduser('~/tmp/labelme_to_coco_preview'),
                            target_folder,options)

path_utils.open_file(html_file)


#%% Back up .json files

backup_folder = os.path.expanduser('~/data/unsw-alting/labeling-backups')
backup_folder += '/{}'.format(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
os.makedirs(backup_folder,exist_ok=True)
commands = []
commands.append('pushd "{}"'.format(labeling_folder_base))
commands.append("find . -name '*.json' | cpio -pdm \"{}\"".format(backup_folder))
commands.append('popd')
commands.append('')

for s in commands:
    print(s)
    
clipboard.copy('\n'.join(commands))
