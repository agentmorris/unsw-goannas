########
#
# unsw-alting-compare-goanna-detections.py
#
# Compare the two goanna detectors (YOLOv5 and YOLOv8), and compare both to the ground truth labels.
#
# Also run on the complete dataset (including unlabeled images) to find previously-missed goannas.
#
########

#%% Imports and constants

import os
import json

from tqdm import tqdm

goanna_detector_yolov5_results_file = os.path.expanduser('~/postprocessing/unsw-alting/unsw-alting-2023-12-10-yolov5-noaug-unknown/combined_api_outputs/unsw-alting-2023-12-10-yolov5-noaug-unknown_detections-threshold.filtered_rde_0.300_0.850_25_0.200.json')
goanna_detector_yolov8_results_file = os.path.expanduser('~/postprocessing/unsw-alting/unsw-alting-2023-12-09-yolov8-noaug-unknown/combined_api_outputs/unsw-alting-2023-12-09-yolov8-noaug-unknown_detections-threshold.filtered_rde_0.300_0.850_25_0.200.json')
megadetector_results_file = os.path.expanduser('~/postprocessing/unsw-alting/unsw-alting-2023-11-02-aug-v5a.0.0/combined_api_outputs/unsw-alting-2023-11-02-aug-v5a.0.0_detections.filtered_rde_0.045_0.850_10_1.010.json')

ground_truth_metadata_folder = os.path.expanduser('~/data/unsw-alting')
ground_truth_coco_file = os.path.join(ground_truth_metadata_folder,'unsw_goannas.json')

image_folder = '/datadrive/home/sftp/unsw-alting_/data'

assert os.path.isfile(goanna_detector_yolov5_results_file)
assert os.path.isfile(goanna_detector_yolov8_results_file)
assert os.path.isfile(megadetector_results_file)

goanna_detector_yolov5_results_file_goannas_only = \
    goanna_detector_yolov5_results_file.replace('.json','_goannas_only.json')
goanna_detector_yolov8_results_file_goannas_only = \
    goanna_detector_yolov8_results_file.replace('.json','_goannas_only.json')

goanna_category_id = '3'


#%% Prepare goanna-only files

with open(goanna_detector_yolov5_results_file,'r') as f:
    goanna_detector_yolov5_results = json.load(f)
assert goanna_detector_yolov5_results['detection_categories'][goanna_category_id] == 'goanna'

with open(goanna_detector_yolov8_results_file,'r') as f:
    goanna_detector_yolov8_results = json.load(f)
assert goanna_detector_yolov8_results['detection_categories'][goanna_category_id] == 'goanna'

from api.batch_processing.postprocessing.subset_json_detector_output import \
    SubsetJsonDetectorOutputOptions,subset_json_detector_output
    
subset_options = SubsetJsonDetectorOutputOptions()
subset_options.categories_to_keep = {goanna_category_id:0.001}
subset_options.overwrite_json_files = True

_ = subset_json_detector_output(goanna_detector_yolov5_results_file, 
                            goanna_detector_yolov5_results_file_goannas_only, 
                            subset_options)

_ = subset_json_detector_output(goanna_detector_yolov8_results_file, 
                            goanna_detector_yolov8_results_file_goannas_only, 
                            subset_options)

    
#%% YOLOv5 / YOLOv8 comparison

from api.batch_processing.postprocessing.compare_batch_results import \
    BatchComparisonOptions, n_way_comparison
    
options = BatchComparisonOptions()

options.parallelize_rendering_with_threads = True

options.job_name = 'goanna-yolo'
options.output_folder = os.path.expanduser('~/tmp/unsw-alting/yolo-comparisons')
options.image_folder = image_folder
options.max_images_per_category = 1000
options.max_images_per_page = None
options.sort_by_confidence = True

options.pairwise_options = []

filenames = [                
    goanna_detector_yolov5_results_file_goannas_only,
    goanna_detector_yolov8_results_file_goannas_only
    ]

detection_thresholds = [0.5,0.5]
rendering_thresholds = None

results = n_way_comparison(filenames,options,detection_thresholds,
                           rendering_thresholds=rendering_thresholds)

from md_utils.path_utils import open_file
open_file(results.html_output_file)


#%% Load ground truth and MD results

with open(ground_truth_coco_file,'r') as f:
    ground_truth = json.load(f)
    
with open(megadetector_results_file,'r') as f:
    md_results = json.load(f)
    
    
#%% Find MD humans/vehicles

md_human_categories = ['2','3']
md_human_threshold = 0.5

md_human_images = []

for im in tqdm(md_results['images']):
    if 'detections' not in im or im['detections'] is None:
        continue
    for det in im['detections']:
        if det['category'] in md_human_categories and det['conf'] >= md_human_threshold:
            md_human_images.append(im['file'])
            break

print('\nFound {} human images (of {})'.format(
    len(md_human_images),len(md_results['images'])))


#%% Find ground truth goannas

gt_goanna_images = []

ground_truth_name_to_id = {c['name']:c['id'] for c in ground_truth['categories']}
ground_truth_goanna_id = ground_truth_name_to_id['goanna']

ground_truth_goanna_image_ids = []

# ann = ground_truth['annotations'][0]
for ann in ground_truth['annotations']:
    if ann['category_id'] == ground_truth_goanna_id:
        ground_truth_goanna_image_ids.append(ann['id'])
        assert ann['id'].endswith('.JPG')
        
print('Found {} goanna images in the ground truth (of {})'.format(
    len(ground_truth_goanna_image_ids),
    len(ground_truth['images'])))


#%% Load goanna model results

with open(goanna_detector_yolov5_results_file_goannas_only,'r') as f:
    goanna_detector_yolov5_results = json.load(f)
    
with open(goanna_detector_yolov8_results_file_goannas_only,'r') as f:
    goanna_detector_yolov8_results = json.load(f)    
    
    
#%% Find predicted goannas

predicted_goanna_files_to_images = {}
goanna_threshold = 0.5

for im in goanna_detector_yolov5_results['images']:
    if im['file'] in predicted_goanna_files_to_images:
        continue
    if 'detections' not in im or im['detections'] is None:
        continue
    for det in im['detections']:
        assert det['category'] == goanna_category_id
        if det['conf'] >= goanna_threshold:
            predicted_goanna_files_to_images[im['file']] = im

for im in goanna_detector_yolov8_results['images']:
    if im['file'] in predicted_goanna_files_to_images:
        continue
    if 'detections' not in im or im['detections'] is None:
        continue    
    for det in im['detections']:
        assert det['category'] == goanna_category_id
        if det['conf'] >= goanna_threshold:
            predicted_goanna_files_to_images[im['file']] = im

predicted_goanna_files = list(predicted_goanna_files_to_images.keys())

print('Predicted goannas in {} files (of {})'.format(
    len(predicted_goanna_files),len(ground_truth['images'])))


#%% Find possibly-new goannas

ground_truth_goanna_image_ids_set = set(ground_truth_goanna_image_ids)
md_human_images_set = set(md_human_images)

possible_new_goannas = []

n_already_labeled = 0
n_human = 0

for fn in predicted_goanna_files:
    if fn in ground_truth_goanna_image_ids_set:
        n_already_labeled += 1
        continue
    if fn in md_human_images_set:
        n_human += 1
        continue
    possible_new_goannas.append(fn)
    
possible_new_goannas_set = set(possible_new_goannas)

print('Found {} possible new goannas ({} already labeled, {} human)'.format(
    len(possible_new_goannas),n_already_labeled,n_human))


#%% Create a new results file with just new goannas

target_folder = os.path.expanduser('~/tmp/unsw-alting/new-goanna-review')
os.makedirs(target_folder,exist_ok=True)
new_goanna_results_file = os.path.join(target_folder,'possible_new_goannas.json')

possible_new_goanna_images = []

for fn in predicted_goanna_files_to_images:
    if fn in possible_new_goannas_set:
        possible_new_goanna_images.append(predicted_goanna_files_to_images[fn])
        
assert len(possible_new_goanna_images) == len(possible_new_goannas_set)

new_goanna_results = {}
new_goanna_results['info'] = goanna_detector_yolov5_results['info']
new_goanna_results['detection_categories'] = goanna_detector_yolov5_results['detection_categories']
new_goanna_results['images'] = possible_new_goanna_images

with open(new_goanna_results_file,'w') as f:
    json.dump(new_goanna_results,f,indent=1)


#%% Render boxes on all the possible new goanna images to a new folder

from md_visualization.visualize_detector_output import visualize_detector_output
    
x = visualize_detector_output(detector_output_path = new_goanna_results_file,
                          out_dir = os.path.join(target_folder,'images'),
                          images_dir = image_folder,
                          confidence_threshold = goanna_threshold,
                          sample = -1,
                          output_image_width = None,
                          random_seed = 0,
                          render_detections_only = False,
                          classification_confidence_threshold = None,
                          html_output_file = os.path.join(target_folder,'index.html'),
                          html_output_options = None,
                          preserve_path_structure = False,
                          parallelize_rendering = True,
                          parallelize_rendering_n_cores = 10,
                          parallelize_rendering_with_threads = True)


#%% Label those images in Timelapse

# ...for goanna presence/absence, and export to .csv.


#%% Load labeled image filenames

timelapse_export_file = os.path.expanduser('~/tmp/unsw-alting/unsw-goanna-review.csv')
assert os.path.isfile(timelapse_export_file)

import pandas as pd
df = pd.read_csv(timelapse_export_file)

goanna_files = []

# i_row = 0; row = df.iloc[i_row]
for i_row,row in tqdm(df.iterrows(),total=len(df)):
    assert row['reviewed']
    if row['goanna']:
        goanna_files.append(row['File'])
        
print('{} of {} images contain goannas'.format(len(goanna_files),len(df)))

with open(megadetector_results_file,'r') as f:
    md_results = json.load(f)

fn_relative_to_results = {im['file']:im for im in md_results['images']}
all_files_relative_set = set(fn_relative_to_results.keys())

#%%

md_threshold = 0.1

n_md_images_above_threshold = 0

# Convert back to original paths
goanna_files_original_paths = []

# i_file = 0; labeled_fn = goanna_files[i_file]
for i_file,labeled_fn in enumerate(goanna_files):
    original_fn_relative = labeled_fn.replace('anno_','').replace('~','/')
    assert original_fn_relative in all_files_relative_set
    goanna_files_original_paths.append(original_fn_relative)
    md_result = fn_relative_to_results[original_fn_relative]
    if len(md_result['detections']) == 0:
        continue
    max_conf = max([det['conf'] for det in md_result['detections']])
    if max_conf >= md_threshold:
        n_md_images_above_threshold += 1

print('Of {} discovered goannas, {} were above a MD threshold of {}'.format(
    len(goanna_files),n_md_images_above_threshold,md_threshold))


#%% Save that list to a new file

new_goannas_file = os.path.expanduser('~/tmp/unsw-alting/new_goannas.txt')
with open(new_goannas_file,'w') as f:
    for fn in goanna_files_original_paths:
        f.write(fn + '\n')