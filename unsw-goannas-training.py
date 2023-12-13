########
#
# unsw-goannas-training.py
#
# This file documents the model training process, starting from where 
# unsw-goannas-prepare-yolo-training-set.py leaves off.  Training happens at the 
# yolo CLI, and the exact command line arguments are documented in the "Train" cell.
#
########

#%% Constants and imports

import os
import shutil
import datetime
import clipboard

training_base_folder = os.path.expanduser('~/tmp/unsw-alting/training')
os.makedirs(training_base_folder,exist_ok=True)

yolo_dataset_file = os.path.expanduser('~/data/unsw-alting/yolo-training-folder/dataset.yml')
assert os.path.isfile(yolo_dataset_file)

import sys
yolov5_dir = os.path.expanduser('~/git/yolov5-current')
if yolov5_dir not in sys.path:
    sys.path.append(yolov5_dir)
    
utils_imported = False
if not utils_imported:
    try:
        from yolov5.utils.general import strip_optimizer
        utils_imported = True
    except Exception:
        pass
if not utils_imported:
    try:
        from ultralytics.utils.general import strip_optimizer # noqa
        utils_imported = True
    except Exception:
        pass        
if not utils_imported:
    try:
        from utils.general import strip_optimizer # noqa
        utils_imported = True
    except Exception:
        pass
assert utils_imported


#%% Environment prep

"""
mamba create --name yolov5 python=3.11 pip -y
cd ~/git
git clone https://github.com/ultralytics/yolov5 yolov5-current
cd yolov5-current
pip install -r requirements.txt
"""

#%% Train (YOLOv5)

"""
cd ~/git/yolov5-current

# I usually have an older commit of yolov5 on my PYTHONPATH, remove it.
export PYTHONPATH=
LD_LIBRARY_PATH=
mamba activate yolov5
"""

batch_size = 8
image_size = 1280
epochs = 300

dt = datetime.datetime.now()
dt_string = '{}{}{}{}{}{}'.format(dt.year,str(dt.month).zfill(2),str(dt.day).zfill(2),
  str(dt.hour).zfill(2),str(dt.minute).zfill(2),str(dt.second).zfill(2))
dt_string = '20231121205829'
assert len(dt_string) == 14

training_run_name = 'unsw-goannas-yolov5-{}-b{}-img{}-e{}'.format(
    dt_string,batch_size,image_size,epochs)

base_model = os.path.expanduser('~/models/camera_traps/megadetector/md_v5.0.0/md_v5a.0.0.pt')
assert os.path.isfile(base_model)

project_dir = training_base_folder

cmd = f'python train.py --img {image_size} --batch {batch_size} --epochs {epochs} --weights "{base_model}" --device 0,1 --project "{project_dir}" --name "{training_run_name}" --data "{yolo_dataset_file}"'

print(cmd)
clipboard.copy(cmd)

# Resume command
resume_checkpoint = os.path.join(project_dir,training_run_name,'weights/last.pt')

# This file doesn't exist when we start training the first time
# os.path.isfile(resume_checkpoint)

resume_cmd = 'python train.py --resume {}'.format(resume_checkpoint)
print(resume_cmd)
# clipboard.copy(resume_cmd)


#%% Train (YOLOv8)

"""
cd ~/git

# I usually have an older commit of yolov5 on my PYTHONPATH, remove it.
export PYTHONPATH=
LD_LIBRARY_PATH=
mamba activate yolov5
"""

batch_size = -1
image_size = 640
epochs = 200

import datetime
dt = datetime.datetime.now()

dt_string = '{}{}{}{}{}{}'.format(dt.year,str(dt.month).zfill(2),str(dt.day).zfill(2),
  str(dt.hour).zfill(2),str(dt.minute).zfill(2),str(dt.second).zfill(2))
dt_string = '20231205053411'
assert len(dt_string) == 14

training_run_name = 'unsw-goannas-yolov8-{}-b{}-img{}-e{}'.format(
    dt_string,batch_size,image_size,epochs)

base_model = 'yolov8x.pt'

if base_model != 'yolov8x.pt':
    assert os.path.isfile(base_model)

project_dir = training_base_folder

cmd = f'mamba activate yolov5 && yolo detect train data="{yolo_dataset_file}" model="{base_model}" epochs={epochs} imgsz={image_size} project="{project_dir}" name="{training_run_name}" device=0,1'

print('Train:\n\n{}'.format(cmd))

clipboard.copy(cmd)

# resume_checkpoint = '/home/user/tmp/unsw-alting/goanna-partial-training/unsw-goannas-transient-yolov8-20231118194424-b-1-img640-e200/weights/last.pt'

# This file doesn't exist yet
resume_checkpoint = os.path.join(project_dir,training_run_name,'weights/last.pt')
resume_checkpoint = '/home/user/models/unsw-goannas/unsw-goannas-yolov8-20231205053411-b-1-img640-e200/20231206/unsw-goannas-yolov8-20231205053411-b-1-img640-e200-last.pt'

resume_command = f'mamba activate yolov5 && yolo detect train resume data="{yolo_dataset_file}" model="{resume_checkpoint}" epochs={epochs} imgsz={image_size} project="{project_dir}" name="{training_run_name}" device=0,1'

print('\nResume:\n\n{}'.format(resume_command))
# print('BACK UP WEIGHTS FIRST'); assert os.path.isfile(resume_checkpoint); clipboard.copy(resume_command)


#%% Back up models after (or during) training, removing optimizer state if appropriate

# Input folder(s)
training_output_dir = os.path.join(project_dir,training_run_name)
training_weights_dir = os.path.join(training_output_dir,'weights')
assert os.path.isdir(training_weights_dir)

# Output folder
model_folder = os.path.expanduser('~/models/unsw-goannas/{}'.format(training_run_name))
checkpoint_tag = '20231210-final'
model_folder = os.path.join(model_folder,checkpoint_tag)
os.makedirs(model_folder,exist_ok=True)

for weight_name in ('last','best'):
    source_file = os.path.join(training_weights_dir,weight_name + '.pt')
    assert os.path.isfile(source_file)
    target_file = os.path.join(model_folder,'{}-{}.pt'.format(
        training_run_name,weight_name))
    
    shutil.copyfile(source_file,target_file)
    target_file_optimizer_stripped = target_file.replace('.pt','-stripped.pt')
    strip_optimizer(target_file,target_file_optimizer_stripped)

other_files = os.listdir(training_output_dir)
other_files = [os.path.join(training_output_dir,fn) for fn in other_files]
other_files = [fn for fn in other_files if os.path.isfile(fn)]

# source_file_abs = other_files[0]
for source_file_abs in other_files:
    assert not source_file_abs.endswith('.pt')
    fn_relative = os.path.basename(source_file_abs)
    target_file_abs = os.path.join(model_folder,fn_relative)
    shutil.copyfile(source_file_abs,target_file_abs)


#%% Make plots during training

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure

from md_utils.path_utils import open_file

assert 'yolov5' in training_run_name or 'yolov8' in training_run_name
if 'yolov5' in training_run_name:
    model_type = 'yolov5'
else:
    model_type = 'yolov8'

results_page_folder = os.path.expanduser('~/tmp/unsw-alting/training/training-progress-report')
os.makedirs(results_page_folder,exist_ok=True)
fig_00_fn_abs = os.path.join(results_page_folder,'figure_00.png')
fig_01_fn_abs = os.path.join(results_page_folder,'figure_01.png')
fig_02_fn_abs = os.path.join(results_page_folder,'figure_02.png')
    
results_file = os.path.expanduser('~/tmp/unsw-alting/training/{}/results.csv'.format(training_run_name))
clipboard.copy(results_file)
assert os.path.isfile(results_file)

df = pd.read_csv(results_file)
df = df.rename(columns=lambda x: x.strip())

plt.ioff()

fig_w = 12
fig_h = 8

fig = matplotlib.figure.Figure(figsize=(fig_w, fig_h), tight_layout=True)
ax = fig.subplots(1, 1)

if model_type == 'yolov5':
    df.plot(x = 'epoch', y = 'val/box_loss', ax = ax) 
    df.plot(x = 'epoch', y = 'val/obj_loss', ax = ax, secondary_y = True) 
    df.plot(x = 'epoch', y = 'train/box_loss', ax = ax) 
    df.plot(x = 'epoch', y = 'train/obj_loss', ax = ax, secondary_y = True) 
else:
    df.plot(x = 'epoch', y = 'val/box_loss', ax = ax) 
    df.plot(x = 'epoch', y = 'val/dfl_loss', ax = ax, secondary_y = True) 
    df.plot(x = 'epoch', y = 'train/box_loss', ax = ax) 
    df.plot(x = 'epoch', y = 'train/dfl_loss', ax = ax, secondary_y = True) 

fig.savefig(fig_00_fn_abs,dpi=100)
plt.close(fig)

fig = matplotlib.figure.Figure(figsize=(fig_w, fig_h), tight_layout=True)
ax = fig.subplots(1, 1)

df.plot(x = 'epoch', y = 'val/cls_loss', ax = ax) 
df.plot(x = 'epoch', y = 'train/cls_loss', ax = ax) 

fig.savefig(fig_01_fn_abs,dpi=100)
plt.close(fig)

fig = matplotlib.figure.Figure(figsize=(fig_w, fig_h), tight_layout=True)
ax = fig.subplots(1, 1)

if model_type == 'yolov5':
    df.plot(x = 'epoch', y = 'metrics/precision', ax = ax)
    df.plot(x = 'epoch', y = 'metrics/recall', ax = ax)
    df.plot(x = 'epoch', y = 'metrics/mAP_0.5', ax = ax)
    df.plot(x = 'epoch', y = 'metrics/mAP_0.5:0.95', ax = ax)
else:
    df.plot(x = 'epoch', y = 'metrics/precision(B)', ax = ax)
    df.plot(x = 'epoch', y = 'metrics/recall(B)', ax = ax)
    df.plot(x = 'epoch', y = 'metrics/mAP50(B)', ax = ax)
    df.plot(x = 'epoch', y = 'metrics/mAP50-95(B)', ax = ax)

fig.savefig(fig_02_fn_abs,dpi=100)
plt.close(fig)

results_page_html_file = os.path.join(results_page_folder,'index.html')
with open(results_page_html_file,'w') as f:
    f.write('<html><body>\n')
    f.write('<img src="figure_00.png"><br/>\n')
    f.write('<img src="figure_01.png"><br/>\n')
    f.write('<img src="figure_02.png"><br/>\n')    
    f.write('</body></html>\n')

open_file(results_page_html_file)
