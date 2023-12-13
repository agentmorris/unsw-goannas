########
#
# manage_local_batch.py
#    
# Semi-automated process for managing a local MegaDetector job, including
# standard postprocessing steps.
#
# This script is not intended to be run from top to bottom like a typical Python script,
# it's a notebook disguised with a .py extension.  It's the Bestest Most Awesome way to
# run MegaDetector, but it's also pretty subtle; if you want to play with this, you might
# want to check in with cameratraps@lila.science for some tips.  Otherwise... YMMV.
#
# Some general notes on using this script, which I do in Spyder, though everything will be
# the same if you are reading this in Jupyter Notebook (using the .ipynb version of the 
# script):
#
# * You can specify the MegaDetector location, but you may find it useful to use the same paths 
#   I use; on all the machines where I run MD, I keep all versions of MegaDetector handy at these 
#   paths:
#  
#   ~/models/camera_traps/megadetector/md_v5.0.0/md_v5a.0.0.pt
#   ~/models/camera_traps/megadetector/md_v5.0.0/md_v5b.0.0.pt
#   ~/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb
#
#   On Windows, this translates to, for example:
#
#   c:\users\dmorr\models\camera_traps\megadetector\md_v5.0.0\md_v5a.0.0.pt
#    
# * Typically when I have a MegaDetector job to run, I make a copy of this script.  Let's 
#   say I'm running a job for an organization called "bibblebop"; I have a big folder of
#   job-specific copies of this script, and I might save a new one called "bibblebop-2023-07-26.py" 
#   (the filename doesn't matter, it just helps me keep these organized).
#
# * There are three variables you need to set in this script before you start running code:
#   "input_path", "organization_name_short", and "job_date".  You will get a sensible error if you forget 
#   to set any of these.  In this case I might set those to "/data/bibblebobcamerastuff",
#   "bibblebop", and "2023-07-26", respectively.
#
# * The defaults assume you want to split the job into two tasks (this is the default because I have 
#   two GPUs).  Nothing bad will happen if you do this on a zero-GPU or single-GPU machine, but if you
#   want everything to run in one logical task, change "n_gpus" and "n_jobs" to 1 (instead of 2).
#
# * After setting the required variables, I run the first few cells - up to and including the one 
#   called "Generate commands" - which collectively take basically zero seconds.  After you run the
#   "Generate commands" cell, you will have a folder that looks something like:
#
#   ~/postprocessing/bibblebop/bibblebop-2023-07-06-mdv5a/
#  
#   On Windows, this means:
#
#   ~/postprocessing/bibblebop/bibblebop-2023-07-06-mdv5a/    
#
#   Everything related to this job - scripts, outputs, intermediate stuff - will be in this folder.
#   Specifically, after the "Generate commands" cell, you'll have scripts in that folder called something
#   like:
#
#   run_chunk_000_gpu_00.sh (or .bat on Windows)
#
#   Personally, I like to run that script directly in a command prompt (I just leave Spyder open, though 
#   it's OK if Spyder gets shut down while MD is running).  
#
#   At this point, once you get the hang of it, you've invested about zero seconds of human time,
#   but possibly several days of unattended compute time, depending on the size of your job.
#   
# * Then when the jobs are done, back to the interactive environment!  I run the next few cells,
#   which make sure the job finished OK, and the cell called "Post-processing (pre-RDE)", which 
#   generates an HTML preview of the results.  You are very plausibly done at this point, and can ignore
#   all the remaining cells.  If you want to do things like repeat detection elimination, or running 
#   a classifier, or splitting your results file up in specialized ways, there are cells for all of those
#   things, but now you're in power-user territory, so I'm going to leave this guide here.  Email
#   cameratraps@lila.science with questions about the fancy stuff.
#
########

#%% Imports and constants

import json
import os
import stat
import time

import humanfriendly

from tqdm import tqdm
from collections import defaultdict

from md_utils import path_utils
from md_utils.ct_utils import is_list_sorted
from md_utils.ct_utils import split_list_into_n_chunks

from detection.run_detector_batch import load_and_run_detector_batch, write_results_to_file
from detection.run_detector import DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD

from api.batch_processing.postprocessing.postprocess_batch_results import (
    PostProcessingOptions, process_batch_results)
from detection.run_detector import get_detector_version_from_filename

max_task_name_length = 92

# To specify a non-default confidence threshold for including detections in the .json file
json_threshold = None

# Turn warnings into errors if more than this many images are missing
max_tolerable_failed_images = 100

use_image_queue = False

# Only relevant when we're using a single GPU
default_gpu_number = 0

quiet_mode = True

# Specify a target image size when running MD... strongly recommended to leave this at "None"
image_size = None

# Only relevant when running on CPU
ncores = 1

# OS-specific script line continuation character
slcc = '\\'

# OS-specific script comment character
scc = '#' 

script_extension = '.sh'

# Prefer threads on Windows, processes on Linux
parallelization_defaults_to_threads = False

# This is for things like image rendering, not for MegaDetector
default_workers_for_parallel_tasks = 30

overwrite_handling = 'skip' # 'skip', 'error', or 'overwrite'

# Set later if EK113/RCNX101-style overflow folders are being handled in this dataset
overflow_folder_handling_enabled = False

if os.name == 'nt':
    slcc = '^'
    scc = 'REM'
    script_extension = '.bat'
    parallelization_defaults_to_threads = True
    default_workers_for_parallel_tasks = 10

## Constants related to using YOLOv5's val.py

# Should we use YOLOv5's val.py instead of run_detector_batch.py?
use_yolo_inference_scripts = True

# Directory in which to run val.py.
yolo_working_dir = os.path.expanduser('~/git/yolov5')

# Only used for loading the mapping from class indices to names
yolo_dataset_file = os.path.expanduser('~/data/unsw-alting/yolo-training-folder/dataset.yml')

# 'yolov5' or 'yolov8'; assumes YOLOv5 if this is None
yolo_model_type = 'yolov8'

# Should we remove intermediate files used for running YOLOv5's val.py?
#
# Only relevant if use_yolo_inference_scripts is True.
remove_yolo_intermediate_results = True
remove_yolo_symlink_folder = True
use_symlinks_for_yolo_inference = True

# Should we apply YOLOv5's test-time augmentation?
augment = False


#%% Constants I set per script

input_path = '/datadrive/home/sftp/unsw-alting_/data'

assert not (input_path.endswith('/') or input_path.endswith('\\'))

organization_name_short = 'unsw-alting'
job_date = '2023-12-09'
assert job_date is not None and organization_name_short != 'organization'

# Optional descriptor
job_tag = 'yolov8-noaug'

if job_tag is None:
    job_description_string = ''
else:
    job_description_string = '-' + job_tag

# model_file = os.path.expanduser('~/models/camera_traps/megadetector/md_v5.0.0/md_v5a.0.0.pt')
# model_file = os.path.expanduser('~/models/camera_traps/megadetector/md_v5.0.0/md_v5b.0.0.pt')
# model_file = os.path.expanduser('~/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb')
model_file = os.path.expanduser('~/models/unsw-goannas/unsw-goannas-yolov8-20231205053411-b-1-img640-e200/20231209-final/unsw-goannas-yolov8-20231205053411-b-1-img640-e200-best.pt')
image_size = 640
yolo_model_type = 'yolov8'

postprocessing_base = os.path.expanduser('~/postprocessing')

# Number of jobs to split data into, typically equal to the number of available GPUs, though
# when using augmentation or an image queue (and thus not using checkpoints), I typically
# use ~100 jobs per GPU; those serve as de facto checkpoints.
n_jobs = 100
n_gpus = 2

# Set to "None" when using augmentation or an image queue, which don't currently support
# checkpointing.  Don't worry, this will be assert()'d in the next cell.
checkpoint_frequency = None

# gpu_images_per_second is only used to print out a time estimate, and it's completely
# tied to the assumption of running on an RTX 3090.  YMMV.
if ('v5') in model_file:
    gpu_images_per_second = 10
else:
    gpu_images_per_second = 2.9
    
# Rough estimate for how much slower everything runs when using augmentation    
if augment:
    gpu_images_per_second = gpu_images_per_second * 0.7
    
base_task_name = organization_name_short + '-' + job_date + job_description_string + '-' + \
    get_detector_version_from_filename(model_file)
base_output_folder_name = os.path.join(postprocessing_base,organization_name_short)
os.makedirs(base_output_folder_name,exist_ok=True)


#%% Derived variables, constant validation, path setup

if use_image_queue:
    assert checkpoint_frequency is None,\
        'Checkpointing is not supported when using an image queue'        
    
if augment:
    assert checkpoint_frequency is None,\
        'Checkpointing is not supported when using augmentation'
    
    assert use_yolo_inference_scripts,\
        'Augmentation is only supported when running with the YOLO inference scripts'

filename_base = os.path.join(base_output_folder_name, base_task_name)
combined_api_output_folder = os.path.join(filename_base, 'combined_api_outputs')
postprocessing_output_folder = os.path.join(filename_base, 'preview')

os.makedirs(filename_base, exist_ok=True)
os.makedirs(combined_api_output_folder, exist_ok=True)
os.makedirs(postprocessing_output_folder, exist_ok=True)

if input_path.endswith('/'):
    input_path = input_path[0:-1]

print('Output folder:\n{}'.format(filename_base))


#%% Enumerate files

all_images = sorted(path_utils.find_images(input_path,recursive=True))

# It's common to run this notebook on an external drive with the main folders in the drive root
all_images = [fn for fn in all_images if not \
              (fn.startswith('$RECYCLE') or fn.startswith('System Volume Information'))]
    
print('Enumerated {} image files in {}'.format(len(all_images),input_path))

if False:

    pass 
    
    #%% Load files from prior enumeration
    
    import re    
    chunk_files = os.listdir(filename_base)
    pattern = re.compile('chunk\d+.json')
    chunk_files = [fn for fn in chunk_files if pattern.match(fn)]
    all_images = []
    for fn in chunk_files:
        with open(os.path.join(filename_base,fn),'r') as f:
            chunk = json.load(f)
            assert isinstance(chunk,list)
            all_images.extend(chunk)
    all_images = sorted(all_images)
    print('Loaded {} image files from chunks in {}'.format(len(all_images),filename_base))
    

#%% Divide images into chunks 

folder_chunks = split_list_into_n_chunks(all_images,n_jobs)


#%% Estimate total time

n_images = len(all_images)
execution_seconds = n_images / gpu_images_per_second
wallclock_seconds = execution_seconds / n_gpus
print('Expected time: {}'.format(humanfriendly.format_timespan(wallclock_seconds)))

seconds_per_chunk = len(folder_chunks[0]) / gpu_images_per_second
print('Expected time per chunk: {}'.format(humanfriendly.format_timespan(seconds_per_chunk)))


#%% Write file lists

task_info = []

for i_chunk,chunk_list in enumerate(folder_chunks):
    
    chunk_fn = os.path.join(filename_base,'chunk{}.json'.format(str(i_chunk).zfill(3)))
    task_info.append({'id':i_chunk,'input_file':chunk_fn})
    path_utils.write_list_to_file(chunk_fn, chunk_list)
    
    
#%% Generate commands

# A list of the scripts tied to each GPU, as absolute paths.  We'll write this out at
# the end so each GPU's list of commands can be run at once.  Generally only used when 
# running lots of small batches via YOLOv5's val.py, which doesn't support checkpointing.
gpu_to_scripts = defaultdict(list)

# i_task = 0; task = task_info[i_task]
for i_task,task in enumerate(task_info):
    
    chunk_file = task['input_file']
    checkpoint_filename = chunk_file.replace('.json','_checkpoint.json')
    
    output_fn = chunk_file.replace('.json','_results.json')
    
    task['output_file'] = output_fn
    
    if n_jobs > 1:
        gpu_number = i_task % n_gpus        
    else:
        gpu_number = default_gpu_number
        
    image_size_string = ''
    if image_size is not None:
        image_size_string = '--image_size {}'.format(image_size)
        
    # Generate the script to run MD
    
    if use_yolo_inference_scripts:

        augment_string = ''
        if augment:
            augment_string = '--augment_enabled 1'
        
        symlink_folder = os.path.join(filename_base,'symlinks','symlinks_{}'.format(
            str(i_task).zfill(3)))
        yolo_results_folder = os.path.join(filename_base,'yolo_results','yolo_results_{}'.format(
            str(i_task).zfill(3)))
                
        symlink_folder_string = '--symlink_folder "{}"'.format(symlink_folder)
        yolo_results_folder_string = '--yolo_results_folder "{}"'.format(yolo_results_folder)
        
        remove_symlink_folder_string = ''
        if not remove_yolo_symlink_folder:
            remove_symlink_folder_string = '--no_remove_symlink_folder'
        
        remove_yolo_results_string = ''
        if not remove_yolo_intermediate_results:
            remove_yolo_results_string = '--no_remove_yolo_results_folder'
        
        confidence_threshold_string = ''
        if json_threshold is not None:
            confidence_threshold_string = '--conf_thres {}'.format(json_threshold)
        else:
            confidence_threshold_string = '--conf_thres {}'.format(DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD)
            
        cmd = ''
        
        device_string = '--device {}'.format(gpu_number)
        
        overwrite_handling_string = '--overwrite_handling {}'.format(overwrite_handling)        
        
        cmd += f'python run_inference_with_yolov5_val.py "{model_file}" "{chunk_file}" "{output_fn}" '
        cmd += f'{image_size_string} {augment_string} '
        cmd += f'{symlink_folder_string} {yolo_results_folder_string} {remove_yolo_results_string} '
        cmd += f'{remove_symlink_folder_string} {confidence_threshold_string} {device_string} '
        cmd += f'{overwrite_handling_string}'
                
        if yolo_working_dir is not None:
            cmd += f' --yolo_working_folder "{yolo_working_dir}"'
        if yolo_dataset_file is not None:
            cmd += ' --yolo_dataset_file "{}"'.format(yolo_dataset_file)
        if yolo_model_type is not None:
            cmd += ' --model_type {}'.format(yolo_model_type)
            
        if not use_symlinks_for_yolo_inference:
            cmd += ' --no_use_symlinks'
        
        cmd += '\n'
        
    else:
        
        if os.name == 'nt':
            cuda_string = f'set CUDA_VISIBLE_DEVICES={gpu_number} & '
        else:
            cuda_string = f'CUDA_VISIBLE_DEVICES={gpu_number} '
                
        checkpoint_frequency_string = ''
        checkpoint_path_string = ''
        
        if checkpoint_frequency is not None and checkpoint_frequency > 0:
            checkpoint_frequency_string = f'--checkpoint_frequency {checkpoint_frequency}'
            checkpoint_path_string = '--checkpoint_path "{}"'.format(checkpoint_filename)
                
        use_image_queue_string = ''
        if (use_image_queue):
            use_image_queue_string = '--use_image_queue'

        ncores_string = ''
        if (ncores > 1):
            ncores_string = '--ncores {}'.format(ncores)
            
        quiet_string = ''
        if quiet_mode:
            quiet_string = '--quiet'
        
        confidence_threshold_string = ''
        if json_threshold is not None:
            confidence_threshold_string = '--threshold {}'.format(json_threshold)
        
        overwrite_handling_string = '--overwrite_handling {}'.format(overwrite_handling)        
        cmd = f'{cuda_string} python run_detector_batch.py "{model_file}" "{chunk_file}" "{output_fn}" {checkpoint_frequency_string} {checkpoint_path_string} {use_image_queue_string} {ncores_string} {quiet_string} {image_size_string} {confidence_threshold_string} {overwrite_handling_string}'
                
    cmd_file = os.path.join(filename_base,'run_chunk_{}_gpu_{}{}'.format(str(i_task).zfill(3),
                            str(gpu_number).zfill(2),script_extension))
    
    with open(cmd_file,'w') as f:
        f.write(cmd + '\n')
    
    st = os.stat(cmd_file)
    os.chmod(cmd_file, st.st_mode | stat.S_IEXEC)
        
    task['command'] = cmd
    task['command_file'] = cmd_file

    # Generate the script to resume from the checkpoint (only supported with MD inference code)
    
    gpu_to_scripts[gpu_number].append(cmd_file)
    
    if checkpoint_frequency is not None:
        
        resume_string = ' --resume_from_checkpoint "{}"'.format(checkpoint_filename)
        resume_cmd = cmd + resume_string
    
        resume_cmd_file = os.path.join(filename_base,
                                       'resume_chunk_{}_gpu_{}{}'.format(str(i_task).zfill(3),
                                       str(gpu_number).zfill(2),script_extension))
        
        with open(resume_cmd_file,'w') as f:
            f.write(resume_cmd + '\n')
        
        st = os.stat(resume_cmd_file)
        os.chmod(resume_cmd_file, st.st_mode | stat.S_IEXEC)
        
        task['resume_command'] = resume_cmd
        task['resume_command_file'] = resume_cmd_file

# ...for each task

# Write out a script for each GPU that runs all of the commands associated with
# that GPU.  Typically only used when running lots of little scripts in lieu
# of checkpointing.
for gpu_number in gpu_to_scripts:
    
    gpu_script_file = os.path.join(filename_base,'run_all_for_gpu_{}{}'.format(
        str(gpu_number).zfill(2),script_extension))
    with open(gpu_script_file,'w') as f:
        for script_name in gpu_to_scripts[gpu_number]:
            s = script_name
            # When calling a series of batch files on Windows from within a batch file, you need to
            # use "call", or only the first will be executed.  No, it doesn't make sense.
            if os.name == 'nt':
                s = 'call ' + s
            f.write(s + '\n')
        f.write('echo "Finished all commands for GPU {}"'.format(gpu_number))
    st = os.stat(gpu_script_file)
    os.chmod(gpu_script_file, st.st_mode | stat.S_IEXEC)

# ...for each GPU


#%% Run the tasks

r"""
The cells we've run so far wrote out some shell scripts (.bat files on Windows, 
.sh files on Linx/Mac) that will run MegaDetector.  I like to leave the interactive
environment at this point and run those scripts at the command line.  So, for example,
if you're on Windows, and you've basically used the default values above, there will be
batch files called, e.g.:

c:\users\[username]\postprocessing\[organization]\[job_name]\run_chunk_000_gpu_00.bat
c:\users\[username]\postprocessing\[organization]\[job_name]\run_chunk_001_gpu_01.bat

Those batch files expect to be run from the "detection" folder of the MegaDetector repo,
typically:
    
c:\git\MegaDetector\detection

All of that said, you don't *have* to do this at the command line.  The following cell 
runs these scripts programmatically, so if you just run the "run the tasks (commented out)"
cell, you should be running MegaDetector.

One downside of the programmatic approach is that this cell doesn't yet parallelize over
multiple processes, so the tasks will run serially.  This only matters if you have multiple
GPUs.
"""

if False:
    
    pass

    #%%% Run the tasks (commented out)

    assert not use_yolo_inference_scripts, \
        'If you want to use the YOLOv5 inference scripts, you can\'t run the model interactively (yet)'
        
    # i_task = 0; task = task_info[i_task]
    for i_task,task in enumerate(task_info):
    
        chunk_file = task['input_file']
        output_fn = task['output_file']
        
        checkpoint_filename = chunk_file.replace('.json','_checkpoint.json')
        
        if json_threshold is not None:
            confidence_threshold = json_threshold
        else:
            confidence_threshold = DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD
            
        if checkpoint_frequency is not None and checkpoint_frequency > 0:
            cp_freq_arg = checkpoint_frequency
        else:
            cp_freq_arg = -1
            
        start_time = time.time()
        results = load_and_run_detector_batch(model_file=model_file, 
                                              image_file_names=chunk_file, 
                                              checkpoint_path=checkpoint_filename, 
                                              confidence_threshold=confidence_threshold,
                                              checkpoint_frequency=cp_freq_arg, 
                                              results=None,
                                              n_cores=ncores, 
                                              use_image_queue=use_image_queue,
                                              quiet=quiet_mode,
                                              image_size=image_size)        
        elapsed = time.time() - start_time
        
        print('Task {}: finished inference for {} images in {}'.format(
            i_task, len(results),humanfriendly.format_timespan(elapsed)))

        # This will write absolute paths to the file, we'll fix this later
        write_results_to_file(results, output_fn, detector_file=model_file)

        if checkpoint_frequency is not None and checkpoint_frequency > 0:
            if os.path.isfile(checkpoint_filename):                
                os.remove(checkpoint_filename)
                print('Deleted checkpoint file {}'.format(checkpoint_filename))
                
    # ...for each chunk
    
# ...if False

    
#%% Load results, look for failed or missing images in each task

n_total_failures = 0

# i_task = 0; task = task_info[i_task]
for i_task,task in enumerate(task_info):
    
    chunk_file = task['input_file']
    output_file = task['output_file']
    
    with open(chunk_file,'r') as f:
        task_images = json.load(f)
    with open(output_file,'r') as f:
        task_results = json.load(f)
    
    task_images_set = set(task_images)
    filename_to_results = {}
    
    n_task_failures = 0
    
    # im = task_results['images'][0]
    for im in task_results['images']:
        assert im['file'].startswith(input_path)
        assert im['file'] in task_images_set
        filename_to_results[im['file']] = im
        if 'failure' in im:
            assert im['failure'] is not None
            n_task_failures += 1
    
    task['n_failures'] = n_task_failures
    task['results'] = task_results
    
    for fn in task_images:
        assert fn in filename_to_results
    
    n_total_failures += n_task_failures

# ...for each task

assert n_total_failures < max_tolerable_failed_images,\
    '{} failures (max tolerable set to {})'.format(n_total_failures,
                                                   max_tolerable_failed_images)

print('Processed all {} images with {} failures'.format(
    len(all_images),n_total_failures))
        

##%% Merge results files and make filenames relative

combined_results = {}
combined_results['images'] = []
images_processed = set()

for i_task,task in enumerate(task_info):

    task_results = task['results']
    
    if i_task == 0:
        combined_results['info'] = task_results['info']
        combined_results['detection_categories'] = task_results['detection_categories']        
    else:
        assert task_results['info']['format_version'] == combined_results['info']['format_version']
        assert task_results['detection_categories'] == combined_results['detection_categories']
        
    # Make sure we didn't see this image in another chunk
    for im in task_results['images']:
        assert im['file'] not in images_processed
        images_processed.add(im['file'])

    combined_results['images'].extend(task_results['images'])
    
# Check that we ended up with the right number of images    
assert len(combined_results['images']) == len(all_images), \
    'Expected {} images in combined results, found {}'.format(
        len(all_images),len(combined_results['images']))

# Check uniqueness
result_filenames = [im['file'] for im in combined_results['images']]
assert len(combined_results['images']) == len(set(result_filenames))

# Check for valid path names
for im in combined_results['images']:
    if input_path.endswith(':'):
        assert im['file'].startswith(input_path)
        im['file'] = im['file'].replace(input_path,'',1)
    else:
        assert im['file'].startswith(input_path + os.path.sep)
        im['file'] = im['file'].replace(input_path + os.path.sep,'',1)
    
combined_api_output_file = os.path.join(
    combined_api_output_folder,
    '{}_detections.json'.format(base_task_name))

with open(combined_api_output_file,'w') as f:
    json.dump(combined_results,f,indent=1)

print('Wrote results to {}'.format(combined_api_output_file))


#%% Remove low-confidence detections

combined_api_output_file_thresholded = combined_api_output_file.replace('.json','-threshold.json')
threshold = 0.01

# im = combined_results['images'][0]
for im in tqdm(combined_results['images']):
    valid_detections = []
    if 'detections' not in im:
        continue
    for d in im['detections']:
        if d['conf'] >= threshold:
            valid_detections.append(d)
    im['detections'] = valid_detections

with open(combined_api_output_file_thresholded,'w') as f:
    json.dump(combined_results,f,indent=1)
    
combined_api_output_file = combined_api_output_file_thresholded


#%% Post-processing (pre-RDE)

render_animals_only = False

options = PostProcessingOptions()
options.image_base_dir = input_path
options.include_almost_detections = True
options.num_images_to_sample = 7500
options.confidence_threshold = 0.2
options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
options.ground_truth_json_file = None
options.separate_detections_by_category = True
options.sample_seed = 0
options.max_figures_per_html_file = 2500

options.parallelize_rendering = True
options.parallelize_rendering_n_cores = default_workers_for_parallel_tasks
options.parallelize_rendering_with_threads = parallelization_defaults_to_threads

if render_animals_only:
    # Omit some pages from the output, useful when animals are rare
    options.rendering_bypass_sets = ['detections_person','detections_vehicle',
                                     'detections_person_vehicle','non_detections']

output_base = os.path.join(postprocessing_output_folder,
    base_task_name + '_{:.3f}'.format(options.confidence_threshold))
if render_animals_only:
    output_base = output_base + '_animals_only'

os.makedirs(output_base, exist_ok=True)
print('Processing to {}'.format(output_base))

options.api_output_file = combined_api_output_file
options.output_dir = output_base
ppresults = process_batch_results(options)
html_output_file = ppresults.output_html_file
path_utils.open_file(html_output_file)


#%% RDE (sample directory collapsing)

#
# The next few cells are about repeat detection elimination; if you want to skip this,
# and still do other stuff in this notebook (e.g. running classifiers), that's fine, but
# the rest of the notebook weakly assumes you've done this.  Specifically, it looks for
# the variable "filtered_api_output_file" (a file produced by the RDE process).  If you
# don't run the RDE cells, just change "filtered_api_output_file" to "combined_api_output_file"
# (the raw output from MegaDetector).  Then it will be like all this RDE stuff doesn't exist.
#
# Though FWIW, once you're sufficiently power-user-ish to use this notebook, RDE is almost
# always worth it.
#

def relative_path_to_location(relative_path):
    """
    This is a sample function that returns a camera name given an image path.  By 
    default in the RDE process, leaf-node folders are equivalent to cameras.  This function
    injects a slightly more sophisticated heuristic that recognizes common overflow folder
    types.
    """
    
    import re
    
    # 100RECNX is the overflow folder style for Reconyx cameras
    # 100EK113 is (for some reason) the overflow folder style for Bushnell cameras
    # 100_BTCF is the overflow folder style for Browning cameras
    # 100MEDIA is the overflow folder style used on a number of consumer-grade cameras
    patterns = ['\/\d+RECNX\/','\/\d+EK\d+\/','\/\d+_BTCF\/','\/\d+MEDIA\/']
    
    relative_path = relative_path.replace('\\','/')    
    for pat in patterns:
        relative_path = re.sub(pat,'/',relative_path)
    location_name = os.path.dirname(relative_path)
    
    return location_name


#%% Test cells for relative_path_to_location

if False:

    pass

    #%% Test the generic cases
    
    relative_path = 'a/b/c/d/100EK113/blah.jpg'
    print(relative_path_to_location(relative_path))
    
    relative_path = 'a/b/c/d/100RECNX/blah.jpg'
    print(relative_path_to_location(relative_path))
    
    
    #%% Test relative_path_to_location on the current dataset
    
    with open(combined_api_output_file,'r') as f:
        d = json.load(f)
    image_filenames = [im['file'] for im in d['images']]
    
    location_names = set()
    
    # relative_path = image_filenames[0]
    for relative_path in tqdm(image_filenames):
        location_name = relative_path_to_location(relative_path)
        location_names.add(location_name)
        
    location_names = list(location_names)
    location_names.sort()
    
    for s in location_names:
        print(s)


#%% Repeat detection elimination, phase 1

# Deliberately leaving these imports here, rather than at the top, because this
# cell is not typically executed
from api.batch_processing.postprocessing.repeat_detection_elimination import repeat_detections_core
task_index = 0

options = repeat_detections_core.RepeatDetectionOptions()

options.confidenceMin = 0.3
options.confidenceMax = 1.01
options.iouThreshold = 0.85
options.occurrenceThreshold = 25
options.maxSuspiciousDetectionSize = 0.2
# options.minSuspiciousDetectionSize = 0.05

options.parallelizationUsesThreads = parallelization_defaults_to_threads
options.nWorkers = default_workers_for_parallel_tasks

# This will cause a very light gray box to get drawn around all the detections
# we're *not* considering as suspicious.
options.bRenderOtherDetections = True
options.otherDetectionsThreshold = options.confidenceMin

options.bRenderDetectionTiles = True
options.maxOutputImageWidth = 2000
options.detectionTilesMaxCrops = 300

# options.lineThickness = 5
# options.boxExpansion = 8

# To invoke custom collapsing of folders for a particular manufacturer's naming scheme
# options.customDirNameFunction = relative_path_to_location; overflow_folder_handling_enabled = True

options.bRenderHtml = False
options.imageBase = input_path
rde_string = 'rde_{:.3f}_{:.3f}_{}_{:.3f}'.format(
    options.confidenceMin, options.iouThreshold,
    options.occurrenceThreshold, options.maxSuspiciousDetectionSize)
options.outputBase = os.path.join(filename_base, rde_string + '_task_{}'.format(task_index))
options.filenameReplacements = None # {'':''}

# Exclude people and vehicles from RDE
# options.excludeClasses = [2,3]

# options.maxImagesPerFolder = 50000
# options.includeFolders = ['a/b/c']
# options.excludeFolder = ['a/b/c']

options.debugMaxDir = -1
options.debugMaxRenderDir = -1
options.debugMaxRenderDetection = -1
options.debugMaxRenderInstance = -1

# Can be None, 'xsort', or 'clustersort'
options.smartSort = 'xsort'

suspicious_detection_results = repeat_detections_core.find_repeat_detections(combined_api_output_file,
                                                                           None,
                                                                           options)


#%% Manual RDE step

## DELETE THE VALID DETECTIONS ##

# If you run this line, it will open the folder up in your file browser
path_utils.open_file(os.path.dirname(suspicious_detection_results.filterFile))

#
# If you ran the previous cell, but then you change your mind and you don't want to do 
# the RDE step, that's fine, but don't just blast through this cell once you've run the 
# previous cell.  If you do that, you're implicitly telling the notebook that you looked 
# at everything in that folder, and confirmed there were no red boxes on animals.
#
# Instead, either change "filtered_api_output_file" below to "combined_api_output_file", 
# or delete *all* the images in the filtering folder.
#


#%% Re-filtering

from api.batch_processing.postprocessing.repeat_detection_elimination import remove_repeat_detections

filtered_output_filename = path_utils.insert_before_extension(combined_api_output_file, 'filtered_{}'.format(rde_string))

remove_repeat_detections.remove_repeat_detections(
    inputFile=combined_api_output_file,
    outputFile=filtered_output_filename,
    filteringDir=os.path.dirname(suspicious_detection_results.filterFile)
    )


#%% Post-processing (post-RDE)

render_animals_only = False

options = PostProcessingOptions()
options.image_base_dir = input_path
options.include_almost_detections = True
options.num_images_to_sample = 7500
options.confidence_threshold = 0.6
options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
options.ground_truth_json_file = None
options.separate_detections_by_category = True
options.sample_seed = 0
options.max_figures_per_html_file = 5000

options.parallelize_rendering = True
options.parallelize_rendering_n_cores = default_workers_for_parallel_tasks
options.parallelize_rendering_with_threads = parallelization_defaults_to_threads

if render_animals_only:
    # Omit some pages from the output, useful when animals are rare
    options.rendering_bypass_sets = ['detections_person','detections_vehicle',
                                      'detections_person_vehicle','non_detections']    

output_base = os.path.join(postprocessing_output_folder, 
    base_task_name + '_{}_{:.3f}'.format(rde_string, options.confidence_threshold))    

if render_animals_only:
    output_base = output_base + '_render_animals_only'
os.makedirs(output_base, exist_ok=True)

print('Processing post-RDE to {}'.format(output_base))

options.api_output_file = filtered_output_filename
options.output_dir = output_base
ppresults = process_batch_results(options)
html_output_file = ppresults.output_html_file

path_utils.open_file(html_output_file)
