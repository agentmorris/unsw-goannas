########
#
# review-usgs-tegu-results.py
#
# Create data review pages for USGS tegu validation data
#
########

#%% Imports and constants

import os

from md_utils.path_utils import open_file
from md_visualization import visualize_db
from data_management.yolo_to_coco import yolo_to_coco
from api.batch_processing.postprocessing.render_detection_confusion_matrix \
    import render_detection_confusion_matrix

# YOLOv5 model
if False:
    model_file = os.path.expanduser('~/models/unsw-goannas/unsw-goannas-yolov5-20231121205829-b8-img1280-e300/20231202/unsw-goannas-yolov5-20231121205829-b8-img1280-e300-best.pt')
    model_type = 'yolov5'
    scratch_folder = os.path.expanduser('~/tmp/unsw-goannas-val-analysis-yolov5')
    confidence_thresholds = {'default':0.5,'goanna':0.4}
    rendering_confidence_thresholds = {'default':0.4,'goanna':0.3}
    job_name = 'unsw-goannas-val-yolov5'; assert ' ' not in job_name

# YOLOv8 model:
if True:
    model_file = os.path.expanduser('~/models/unsw-goannas/unsw-goannas-yolov8-20231205053411-b-1-img640-e200/20231209-final/unsw-goannas-yolov8-20231205053411-b-1-img640-e200-best.pt')
    model_type = 'yolov8'
    scratch_folder = os.path.expanduser('~/tmp/unsw-goannas-val-analysis-yolov8')
    confidence_thresholds = {'default':0.5,'goanna':0.4}
    rendering_confidence_thresholds = {'default':0.4,'goanna':0.3}
    job_name = 'unsw-goannas-val-yolov8'; assert ' ' not in job_name

assert os.path.isfile(model_file)

augment = True

if augment:
    job_name += '-aug'
    scratch_folder += '-aug'
    
os.makedirs(scratch_folder,exist_ok=True)

if model_type == 'yolov5':
    yolo_working_folder = os.path.expanduser('~/git/yolov5-current')
else:
    yolo_working_folder = None

# This is the top-level folder, with 'train' and 'val' subfolders    
training_data_folder = os.path.expanduser('~/data/unsw-alting/yolo-training-folder/')
assert os.path.isdir(training_data_folder)

val_image_folder = os.path.join(training_data_folder,'val')
assert os.path.isdir(val_image_folder)

yolo_dataset_file = os.path.expanduser('~/data/unsw-alting/yolo-training-folder/dataset.yml')
assert os.path.isfile(yolo_dataset_file)

# This doesn't exist yet, we'll create this later
training_metadata_file = os.path.expanduser('~/data/unsw-alting/unsw-goannas-training.json')
training_metadata_file_val_only = \
    training_metadata_file.replace('.json','-val_only.json')
assert training_metadata_file_val_only != training_metadata_file    

results_file = os.path.join(scratch_folder,'{}_val_results.json'.format(job_name))

preview_folder = os.path.join(scratch_folder,'preview')
preview_images_folder = os.path.join(preview_folder,'images')
os.makedirs(preview_images_folder,exist_ok=True)

target_image_size = (1280,-1)

parallelize_rendering = True
parallelize_rendering_n_cores = 10
parallelize_rendering_with_threads = False

force_render_images = False


#%% Run the model on the validation data

cmd = 'mamba activate yolov5 && cd ~/git/MegaDetector/detection && export PYTHONPATH=/home/user/git/MegaDetector'
cmd += ' && python run_inference_with_yolov5_val.py "{}" "{}" "{}" --model_type {}'.format(
    model_file,
    val_image_folder,
    results_file,
    model_type)

if model_type == 'yolov5':
    cmd += ' --yolo_working_folder {}'.format(yolo_working_folder)
    
cmd += ' --overwrite_handling overwrite'
cmd += ' --yolo_dataset_file {}'.format(yolo_dataset_file)

if not augment:
    cmd += ' --augment_enabled 0'

print(cmd)
# import clipboard; clipboard.copy(cmd)


#%% Create a CCT .json file for the validation data

coco_formatted_val_data = yolo_to_coco(input_folder=val_image_folder,
                                       class_name_file=yolo_dataset_file,
                                       output_file=training_metadata_file_val_only)


#%% Preview the .json file

options = visualize_db.DbVizOptions()
options.parallelize_rendering = True
options.viz_size = (900, -1)
options.num_to_visualize = 5000

html_file,_ = visualize_db.visualize_db(training_metadata_file_val_only,
                                        os.path.join(scratch_folder,'yolo_to_coco_preview'),
                                        val_image_folder,options)    

open_file(html_file)


#%% Render a confusion matrix

ground_truth_file = training_metadata_file_val_only
image_folder = val_image_folder
empty_category_name = 'empty'

html_image_list_options = {'maxFiguresPerHtmlFile':3000}

confusion_matrix_info = render_detection_confusion_matrix(ground_truth_file=ground_truth_file,
                                  results_file=results_file,
                                  image_folder=image_folder,
                                  preview_folder=preview_folder,
                                  force_render_images=force_render_images,
                                  confidence_thresholds=confidence_thresholds,
                                  rendering_confidence_thresholds=rendering_confidence_thresholds,
                                  target_image_size=target_image_size,
                                  parallelize_rendering=parallelize_rendering,
                                  parallelize_rendering_n_cores=parallelize_rendering_n_cores,
                                  parallelize_rendering_with_threads=parallelize_rendering_with_threads,
                                  job_name=job_name,
                                  model_file=model_file,
                                  html_image_list_options=html_image_list_options)

open_file(confusion_matrix_info['html_file'])
