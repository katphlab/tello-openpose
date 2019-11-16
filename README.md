# tello-openpose
This repo replaces [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) framework used in [tello-openpose](https://github.com/katphlab/tello-openpose) repository with a [lighter and faster implementation of pose estimation](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch) which can even run on CPU using [OpenVINO](https://software.intel.com/en-us/openvino-toolkit).

**Majority of the credit goes to owners of the above repositories**, some credit to **me** for integrating the new pose detection :)

Tested only on Ubuntu 18.04 using MX150 (2GB) graphics card giving 15-20 FPS. If you want to use a CPU, install OpenVINO SDK and change the inference method.

**TODO**
1. Make the code easier to use and understand 
2. Add requirements.txt 

## Libraries and packages
I've attached a conda environent for the versions of the packages, use it at with caution.

### OpenCV, pynput, pygame : 
Mainly used for the UI (display windows, read keyboard events, play sounds). Any recent version should work.

### Pose estimation :
I use the official release https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch

Download the pretrained model from the [original author's link](https://drive.google.com/file/d/1niBUbUecPhKt3GyeDNukobL4OQ3jqssH/view?usp=sharing) and place it in the base folder.

### TelloPy :
Python package which controls the Tello drone.
https://github.com/hanyazou/TelloPy
Do not install from pip, clone the repo and install from there, else some functionality might not work.

### simple-pid :
A simple and easy to use PID controller in Python.
https://github.com/m-lundberg/simple-pid

Used here to control the yaw, pitch, rolling and throttle of the drone. 

The parameters of the PIDs may depend on the processing speed and need tuning to adapt to the FPS you can get. For instance, if the PID that controls the yaw works well at 20 frames/sec, it may yield oscillating yaw at 10 frames/sec.  

## Files of the repository

### pose_wrapper.py :
My own layer using the pose estimation demo: https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch/blob/master/demo.py

Change the model path in pose_wrapper if changing the model.

### CameraMorse.py :

Designed with the Tello drone in mind but could be used with other small cameras.
When the Tello drone is not flying, we can use its camera as a way to pass commands to the calling script.
Covering/uncovering the camera with a finger, is like pressing/releasing a button. 
Covering/uncovering the camera is determined by calculating the level of brightness of the frames received from the camera
Short press = dot
Long press = dash
If we associate series of dots/dashes to commands, we can then ask the script to launch these commands.

Look at the main of the file to have an example.

### SoundPlayer.py :
Defines classes SoundPlayer and Tone used to play sounds and sine wave tones in order to give a audio feedback to the user, for instance, when a pose is recognized or when using CameraMorse. Based on pygame. The sounds are ogg files in 'sounds' directory.

### FPS.py :
Defines class FPS, which calculates and displays the current frames/second.

### tello_openpose.py :
This is the main application. 

Instead of starting from scratch, I used the code from: https://github.com/Ubotica/telloCV/blob/master/telloCV.py A lot of modifications since then, but the hud still looks similar.

Just run with:
python tello_openpose.py

-----
A big thanks to all the people who wrote and shared the libraries/programs I have used for this project !





