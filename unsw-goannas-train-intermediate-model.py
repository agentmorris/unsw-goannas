########
#
# unsw-goannas-train-intermediate-model.py
#
# Do a one-off train/val split and train a YOLOv5 model for a partially-finished
# labelme folder.
#
########

#%% Constants and imports

from collections import defaultdict

from data_management import labelme_to_yolo
from md_utils.path_utils import recursive_file_list
from md_utils.path_utils import find_images
from md_utils.path_utils import safe_create_link

import os
import json
import shutil
import clipboard

input_folder = os.path.expanduser('~/data/unsw-alting/labelme-folders/goanna')
training_base_folder = os.path.expanduser('~/tmp/unsw-alting/goanna-partial-training')

yolo_dataset_file = os.path.join(training_base_folder,'dataset.yaml')

assert os.path.isdir(input_folder)
os.makedirs(training_base_folder,exist_ok=True)


#%% Find all the files that have been labeled

all_label_files = recursive_file_list(input_folder,return_relative_paths=True)
all_label_files = [s for s in all_label_files if s.endswith('.json')]
all_label_files = [s for s in all_label_files if 'alt.json' not in s]

completed_label_files = []

for fn_relative in all_label_files:
    fn_abs = os.path.join(input_folder,fn_relative)
    with open(fn_abs,'r') as f:
        d = json.load(f)
        if 'saved_by_labelme' in d:
            completed_label_files.append(fn_relative)
        
print('{} of {} files have been reviewed'.format(
    len(completed_label_files),len(all_label_files)))


#%% Map to locations

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

for fn_relative in completed_label_files:
    location_id = fn_to_location(fn_relative)
    location_to_files[location_id].append(fn_relative)
    
location_to_files = {k: v for k, v in sorted(
    location_to_files.items(), key=lambda item: len(item[1]))}    

for location_id in location_to_files:
    print('{}: {}'.format(location_id,len(location_to_files[location_id])))
    
    
#%% Split locations into train/val

import random
target_val_fraction = 0.2
max_random_seed = 1000

location_ids = list(location_to_files.keys())
n_val_locations = int(target_val_fraction*len(location_ids))

random_seed_to_error = {}

total_images = sum([len(x) for x in location_to_files.values()])

def compute_seed_error(random_seed):
    
    # Randomly split into train/val
    random.seed(random_seed)
    val_locations = random.sample(location_ids,k=n_val_locations)
    
    n_val_images = 0
    for loc in val_locations:
        n_val_images += len(location_to_files[loc])
    
    val_fraction = n_val_images / total_images
    seed_error = abs(val_fraction-target_val_fraction)
    return seed_error,val_locations

min_error = None
min_error_seed = None

for seed in range(0,max_random_seed):
    seed_error,_ = compute_seed_error(seed)
    if min_error is None or seed_error < min_error:
        min_error = seed_error
        min_error_seed = seed    

print('Min error: {} ({})'.format(min_error,min_error_seed))

_,val_locations = compute_seed_error(min_error_seed)

train_locations = [loc for loc in location_ids if loc not in val_locations]


#%% Prepare YOLO .txt files

yolo_category_name_to_category_id = labelme_to_yolo.labelme_folder_to_yolo(
                                        labelme_folder=input_folder,
                                        category_name_to_category_id=None,
                                        required_token='saved_by_labelme',
                                        right_edge_quantization_threshold=0.015,
                                        overwrite_behavior='overwrite')


#%% Find the image corresponding to each label file

image_files = find_images(input_folder,return_relative_paths=True)
image_file_base_to_image = {}
for fn_relative in image_files:
    image_file_base_to_image[os.path.splitext(fn_relative)[0]] = fn_relative

label_file_to_image_file = {}

# fn_relative = completed_label_files[0]
for fn_relative in completed_label_files:
    bn = os.path.splitext(fn_relative)[0]
    assert bn in image_file_base_to_image
    label_file_to_image_file[fn_relative] = image_file_base_to_image[bn]

assert len(label_file_to_image_file) == len(completed_label_files)


#%% Create symlinks to images and json files

train_image_dir = os.path.join(training_base_folder,'train')
val_image_dir = os.path.join(training_base_folder,'val')

shutil.rmtree(train_image_dir)
shutil.rmtree(val_image_dir)

os.makedirs(train_image_dir,exist_ok=True)
os.makedirs(val_image_dir,exist_ok=True)

# label_fn_relative = completed_label_files[0]
for label_fn_relative in completed_label_files:
    
    location_id = fn_to_location(label_fn_relative)
    assert location_id in val_locations or location_id in train_locations
        
    if location_id in val_locations:
        target_folder = val_image_dir
    else:
        target_folder = train_image_dir
        
    image_file_relative = label_file_to_image_file[label_fn_relative]
    yolo_file_relative = os.path.splitext(label_fn_relative)[0] + '.txt'
    
    image_file_abs = os.path.join(input_folder,image_file_relative)
    yolo_file_abs = os.path.join(input_folder,yolo_file_relative)
    
    assert os.path.isfile(image_file_abs) and os.path.isfile(yolo_file_abs)
    
    target_image_file_abs = os.path.join(target_folder,image_file_relative).replace('.JPG','.jpg')
    target_yolo_file_abs = os.path.join(target_folder,yolo_file_relative)
    
    target_dir = os.path.dirname(target_image_file_abs)
    assert os.path.isdir(target_dir)
    
    safe_create_link(image_file_abs,target_image_file_abs)
    # os.symlink(image_file_abs,target_image_file_abs)
    safe_create_link(yolo_file_abs,target_yolo_file_abs)
    
# ...for each file


#%% Create metadata files

## Create class list files in each folder for BoundingBoxEditor

yolo_category_name_to_category_id = \
    {k: v for k, v in sorted(yolo_category_name_to_category_id.items(), key=lambda item: item[1])}
    
from md_utils.ct_utils import list_is_sorted
assert list_is_sorted(list(yolo_category_name_to_category_id.values()))

# image_dir = val_image_dir
for image_dir in (train_image_dir,val_image_dir):
    output_file = os.path.join(image_dir,'object.data')
    with open(output_file,'w') as f:
        for category_name in yolo_category_name_to_category_id.keys():
            f.write(category_name + '\n')
    
## Create the YOLO dataset.yaml file

from data_management.coco_to_yolo import write_yolo_dataset_file

write_yolo_dataset_file(yolo_dataset_file,
                        training_base_folder,
                        list(yolo_category_name_to_category_id.keys()),
                        train_folder_relative='train',
                        val_folder_relative='val',
                        test_folder_relative=None)


#%% Train (YOLOv5)

"""
cd ~/git/yolov5-current

# I usually have an older commit of yolov5 on my PYTHONPATH, remove it.
export PYTHONPATH=
LD_LIBRARY_PATH=
mamba activate yolov5
"""

# On my 2x24GB GPU setup, a batch size of 16 failed, but 8 was safe.  Autobatch did not
# work; I got an incomprehensible error that I decided not to fix, but I'm pretty sure
# it would have come out with a batch size of 8 anyway.
batch_size = 8
image_size = 1280
epochs = 100

import datetime
dt = datetime.datetime.now()
dt.year
training_run_name = 'unsw-goannas-transient-yolov5-{}{}{}{}{}{}-b{}-img{}-e{}'.format(
    dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second,batch_size,image_size,epochs)

# base_model = os.path.expanduser('~/models/camera_traps/megadetector/md_v5.0.0/md_v5a.0.0.pt')
base_model = '/home/user/models/unsw-goannas/unsw-goannas-transient-20231112-b8-img1280-e100/unsw-goannas-transient-20231112-b8-img1280-e100-best.pt'

assert os.path.isfile(base_model)

project_dir = training_base_folder

cmd = f'python train.py --img {image_size} --batch {batch_size} --epochs {epochs} --weights "{base_model}" --device 0,1 --project {project_dir} --name {training_run_name} --data "{yolo_dataset_file}"'

print(cmd)
clipboard.copy(cmd)

    
#%% Train (YOLOv8)

"""
cd ~/git

# I usually have an older commit of yolov5 on my PYTHONPATH, remove it.
export PYTHONPATH=
LD_LIBRARY_PATH=
mamba activate yolov5
"""

batch_size = -1
image_size = 640 # 1280
epochs = 200

import datetime
dt = datetime.datetime.now()
dt.year
training_run_name = 'unsw-goannas-transient-yolov8-{}{}{}{}{}{}-b{}-img{}-e{}'.format(
    dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second,batch_size,image_size,epochs)

base_model = 'yolov8x.pt'
# base_model = '/home/user/models/usgs-tegus/usgs-tegus-yolov8x-2023.10.26-b-1-img640-e300-best.pt'
# base_model = '/home/user/models/unsw-goannas/unsw-goannas-transient-yolov8-2023111713648-b-1-img640-e200/unsw-goannas-transient-yolov8-2023111713648-b-1-img640-e200-best.pt'

if base_model != 'yolov8x.pt':
    assert os.path.isfile(base_model)

project_dir = training_base_folder

cmd = f'mamba activate yolov5 && yolo detect train data="{yolo_dataset_file}" model="{base_model}" epochs={epochs} imgsz={image_size} project="{project_dir}" name="{training_run_name}" device=0,1'

print('Train:\n\n{}'.format(cmd))

clipboard.copy(cmd)

# resume_checkpoint = '/home/user/tmp/unsw-alting/goanna-partial-training/unsw-goannas-transient-yolov8-20231118194424-b-1-img640-e200/weights/last.pt'

# This file doesn't exist yet
resume_checkpoint = os.path.join(project_dir,training_run_name,'weights/last.pt')

# assert os.path.isfile(resume_checkpoint)

resume_command = f'mamba activate yolov5 && yolo detect train data="{yolo_dataset_file}" model="{resume_checkpoint}" epochs={epochs} imgsz={image_size} project="{project_dir}" name="{training_run_name}" device=0,1 resume'

print('\nResume:\n\n{}'.format(resume_command))
# clipboard.copy(resume_command)


#%% Back up models after training

model_folder = os.path.expanduser('~/models/unsw-goannas/{}'.format(training_run_name))
os.makedirs(model_folder,exist_ok=True)

training_output_dir = os.path.join(project_dir,training_run_name,'weights')
assert os.path.isdir(training_output_dir)

for weight_name in ('last','best'):
    source_file = os.path.join(training_output_dir,weight_name + '.pt')
    assert os.path.isfile(source_file)
    target_file = os.path.join(model_folder,'{}-{}.pt'.format(
        training_run_name,weight_name))
    shutil.copyfile(source_file,target_file)

assert 'best' in target_file
clipboard.copy(target_file)


#%% Run inference on validation data

# model_file = '/home/user/models/unsw-goannas/unsw-goannas-transient-20231112-b8-img1280-e100/unsw-goannas-transient-20231112-b8-img1280-e100-best.pt'

# model_file = '/home/user/models/unsw-goannas/unsw-goannas-transient-yolov8-20231114-b-1-img640-e200/unsw-goannas-transient-yolov8-20231114-b-1-img640-e200-best.pt'

# model_file = '/home/user/tmp/unsw-alting/goanna-partial-training/unsw-goannas-transient-yolov8-2023111519262-b-1-img1280-e200/weights/best.pt'

# model_file = '/home/user/models/unsw-goannas/unsw-goannas-transient-yolov8-2023111620736-b-1-img640-e200/unsw-goannas-transient-yolov8-2023111620736-b-1-img640-e200-best.pt'

model_file = '/home/user/models/unsw-goannas/unsw-goannas-transient-yolov8-2023111713648-b-1-img640-e200/unsw-goannas-transient-yolov8-2023111713648-b-1-img640-e200-best.pt'

image_size_str = model_file.split('img')[1].split('-')[0]; _ = int(image_size_str)

input_folder = val_image_dir
inference_output_file = os.path.expanduser('~/tmp/unsw-alting/inference-results/val-{}/val-{}.json'.format(
    training_run_name,training_run_name))

print(model_file)
print(input_folder)
print(inference_output_file)

if 'yolov8' in model_file:
    cmd = 'mamba activate yolov5 && python run_inference_with_yolov5_val.py "{}" "{}" "{}"'.format(model_file,input_folder,inference_output_file)
    cmd += ' --image_size {}'.format(image_size_str)
    cmd += ' --model_type yolov8'
else:
    cmd = 'mamba activate yolov5 && export PYTHONPATH=/home/user/git/MegaDetector:/home/user/git/yolov5-current && python run_detector_batch.py "{}" "{}" "{}" --recursive --output_relative_filenames --image_size {} --quiet'.format(
        model_file,input_folder,inference_output_file,image_size_str)

print(cmd)
clipboard.copy(cmd)


#%% Preview results on validation data

val_preview_dir = os.path.expanduser('~/tmp/unsw-alting/inference-results/val-{}/preview'.format(
    training_run_name))

cmd = 'python ../api/batch_processing/postprocessing/postprocess_batch_results.py "{}" "{}"'.format(
    inference_output_file,val_preview_dir)
cmd += ' --image_base_dir "{}"'.format(val_image_dir)
cmd += ' --include_almost_detections --num_images_to_sample -1 --n_cores 10 --open_output_file --confidence_threshold 0.1 --almost_detection_confidence_threshold 0.05'

print(cmd)
clipboard.copy(cmd)

# from md_utils.path_utils import open_file; open_file(os.path.join(val_preview_dir,'index.html'))


#%% Run inference on the entire "goanna" folder

# model_file = '/home/user/models/unsw-goannas/unsw-goannas-transient-20231112-b8-img1280-e100/unsw-goannas-transient-20231112-b8-img1280-e100-best.pt'; image_size = 1280; model_type = 'yolov5'

# model_file = '/home/user/models/unsw-goannas/unsw-goannas-transient-yolov8-20231114-b-1-img640-e200/unsw-goannas-transient-yolov8-20231114-b-1-img640-e200-best.pt'; image_size = 640; model_type = 'yolov8'

# model_file = '/home/user/models/unsw-goannas/unsw-goannas-transient-yolov8-20231114195042-b-1-img640-e200/unsw-goannas-transient-yolov8-20231114195042-b-1-img640-e200-best.pt'; model_type = 'yolov8'

# model_file = '/home/user/tmp/unsw-alting/goanna-partial-training/unsw-goannas-transient-yolov8-2023111519262-b-1-img1280-e200/weights/best.pt'; model_type = 'yolov8'

# model_file = '/home/user/models/unsw-goannas/unsw-goannas-transient-yolov8-2023111620736-b-1-img640-e200/unsw-goannas-transient-yolov8-2023111620736-b-1-img640-e200-best.pt'; model_type = 'yolov8'

# model_file = '/home/user/models/unsw-goannas/unsw-goannas-transient-yolov8-2023111713648-b-1-img640-e200/unsw-goannas-transient-yolov8-2023111713648-b-1-img640-e200-best.pt'

model_file = '/home/user/models/unsw-goannas/unsw-goannas-transient-yolov8-20231118194424-b-1-img640-e200/unsw-goannas-transient-yolov8-20231118194424-b-1-img640-e200-best.pt'

assert 'yolov5' in model_file or 'yolov8' in model_file
if 'yolov5' in model_file:
    model_type = 'yolov5'
else:
    model_type = 'yolov8'
    
image_size_str = model_file.split('img')[1].split('-')[0]; _ = int(image_size_str)

input_folder = '/home/user/data/unsw-alting/labelme-folders/goanna'

augment = False

if augment:
    aug_string = '-aug'
    augment_value = 1
else:
    aug_string = ''
    augment_value = 0
    
inference_output_file = os.path.expanduser(
    '~/tmp/unsw-alting/inference-results/all-goannas-{}/all-goannas-{}{}.json'.format(
    training_run_name,training_run_name,aug_string))

cmd = 'mamba activate yolov5 && cd ~/git/MegaDetector/detection && export PYTHONPATH=/home/user/git/MegaDetector:/home/user/git/yolov5-current && python run_inference_with_yolov5_val.py "{}" "{}" "{}" --augment {} --image_size {} --model_type {}'.format(
    model_file,input_folder,inference_output_file,augment_value,image_size_str,model_type)
cmd += ' --yolo_working_folder "{}"'.format('/home/user/git/yolov5-current')

print(cmd)
clipboard.copy(cmd)


#%% Preview results on the entire "goanna" folder

all_goanna_preview_dir = os.path.expanduser('~/tmp/unsw-alting/inference-results/all-goannas-{}/preview'.format(
    training_run_name))

cmd = 'python ../api/batch_processing/postprocessing/postprocess_batch_results.py "{}" "{}"'.format(
    inference_output_file,all_goanna_preview_dir)
cmd += ' --image_base_dir "{}"'.format(input_folder)
cmd += ' --include_almost_detections --num_images_to_sample 10000 --n_cores 10 --open_output_file --confidence_threshold 0.1 --almost_detection_confidence_threshold 0.05 --max_figures_per_html_file 2500'

print(cmd)
clipboard.copy(cmd)

# from md_utils.path_utils import open_file; open_file(os.path.join(val_preview_dir,'index.html'))


#%% Compare results (e.g. with/without aug)

results_file_a = '/home/user/tmp/unsw-alting/inference-results/all-goannas-unsw-goannas-transient-yolov8-2023111713648-b-1-img640-e200/all-goannas-unsw-goannas-transient-yolov8-2023111713648-b-1-img640-e200-aug.json'

results_file_b = '/home/user/tmp/unsw-alting/inference-results/all-goannas-unsw-goannas-transient-yolov8-2023111713648-b-1-img640-e200/all-goannas-unsw-goannas-transient-yolov8-2023111713648-b-1-img640-e200.json'

assert results_file_a != results_file_b
assert os.path.isfile(results_file_a)
assert os.path.isfile(results_file_b)

import itertools

from api.batch_processing.postprocessing.compare_batch_results import (
    BatchComparisonOptions,PairwiseBatchComparisonOptions,compare_batch_results)

options = BatchComparisonOptions()

all_goanna_comparison_dir = os.path.expanduser('~/tmp/unsw-alting/inference-results/all-goannas-{}/comparison'.format(
    training_run_name))

options.job_name = 'gonna-aug-comparison'
options.output_folder = all_goanna_comparison_dir
options.image_folder = input_folder

options.pairwise_options = []

filenames = [results_file_a,results_file_b]    

detection_thresholds = [0.1,0.1]

assert len(detection_thresholds) == len(filenames)

rendering_thresholds = [(x*0.6666) for x in detection_thresholds]

# Choose all pairwise combinations of the files in [filenames]
for i, j in itertools.combinations(list(range(0,len(filenames))),2):
        
    pairwise_options = PairwiseBatchComparisonOptions()
    
    pairwise_options.results_filename_a = filenames[i]
    pairwise_options.results_filename_b = filenames[j]
    
    pairwise_options.rendering_confidence_threshold_a = rendering_thresholds[i]
    pairwise_options.rendering_confidence_threshold_b = rendering_thresholds[j]
    
    pairwise_options.detection_thresholds_a = {'animal':detection_thresholds[i],
                                               'person':detection_thresholds[i],
                                               'vehicle':detection_thresholds[i]}
    pairwise_options.detection_thresholds_b = {'animal':detection_thresholds[j],
                                               'person':detection_thresholds[j],
                                               'vehicle':detection_thresholds[j]}
    options.pairwise_options.append(pairwise_options)

results = compare_batch_results(options)

from md_utils.path_utils import open_file
open_file(results.html_output_file)


#%% Run inference on one image

"""
yolo predict model="/home/user/models/unsw-goannas/unsw-goannas-transient-yolov8-20231114-b-1-img640-e200/unsw-goannas-transient-yolov8-20231114-b-1-img640-e200-best.pt" source="/home/user/tmp/labels/BrendanAltingMLDP2023Images#Q16#Q16__2023-02-15__14-11-54(17).JPG" imgsz=640 save=True project="/home/user/tmp/labels" name="test" augment
"""
