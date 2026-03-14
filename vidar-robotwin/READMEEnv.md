## Python Env
Create ENV
```
conda env create --file RoboTwin-hb.yaml
conda activate RoboTwin-hb
```
Install additional packages
```
bash script/_install.sh
```
Download assets
```
bash script/_download_assets.sh
```
[Fix oidn](https://github.com/RoboTwin-Platform/RoboTwin/issues/52). Check $CONDA_PREFIX first, then run
```
bash script/_fix_oidn.sh
```

## Robot Env
### Instructions
#### Tasks using the left arm
```
The whole scene is in a realistic, industrial art style with three views: a fixed rear camera, a movable left arm camera, and a movable right arm camera. The aloha robot is currently performing the following task: using left arm, [description]
```
#### Tasks using the right arm
```
The whole scene is in a realistic, industrial art style with three views: a fixed rear camera, a movable left arm camera, and a movable right arm camera. The aloha robot is currently performing the following task: using right arm, [description]
```
#### Tasks using both arms
```
The whole scene is in a realistic, industrial art style with three views: a fixed rear camera, a movable left arm camera, and a movable right arm camera. The aloha robot is currently performing the following task: [description]
```
[description] means the "full_description" item in ./description/task_instruction/TASK_NAME.json

#### Special Tasks
For place_a2b_left, place_a2b_right, and put_object_cabinet, we use RoboTwin-generated seen descriptions because of the placeholder issue
```
The whole scene is in a realistic, industrial art style with three views: a fixed rear camera, a movable left arm camera, and a movable right arm camera. The aloha robot is currently performing the following task: [RoboTwin-generated seen description]
```
