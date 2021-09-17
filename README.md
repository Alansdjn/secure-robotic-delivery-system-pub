# **Handbook**

## **Project Title:**

**An AI-driven Secure and Intelligent Robotic Delivery System**

## **Please see Code files in GitHub** :

[https://github.com/Alansdjn/secure-robotic-delivery-system-pub](https://github.com/Alansdjn/secure-robotic-delivery-system-pub)

## **Brief Introduction:**

**This document is divided into three parts:**

1. Explanation of files in project _An AI-driven Secure and Intelligent Robotic Delivery System_.

2. Guidance of simple experiment.

3. Guidance of system implementation (demo).

**Datasets:**
[TIMIT](https://github.com/philipperemy/timit)
[Speech Commands](https://arxiv.org/abs/1804.03209)
[CASIA-WebFace](https://arxiv.org/pdf/1411.7923v1.pdf)
[Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/)

Fig. 1 and Fig. 2 show how the system work. For more details please read the project&#39;s paper.

![](https://github.com/Alansdjn/secure-robotic-delivery-system-pub/blob/main/images/system_model.png)
Figure 1. The proposed robotic delivery system

![](https://github.com/Alansdjn/secure-robotic-delivery-system-pub/blob/main/images/The_offline_authentication_approaches_of_the_proposed_system.png)
Figure 2. Details of off-line authentication scheme

## **1. Explanation of files**

### A. cnn_xgboost

This folder is for PIN code recogination in the cooperative authentication of our system.

**generate_train_data.py**: This file generates the pre-converted trainging images for CNN model.
**train_cnn.py**: is used for training the CNN model.
**tune_xgboost_ray.py**: We use Tune Ray framework to tune the hyperparameters of XGBoost classifier.

**core**: This folder contains the basic functions which are used by _train_cnn.py_ and _tune_xgboost_ray.py_ and _generate_train_data.py_. 
- **model.py** defines the CNN model, 
- **dataset.py** contains the definition of dataloader, and 
- **utils.py** is the implementations of util functions.

**HPC**: The bash files in this folder is the scripts used for submit jobs which are runned on the High Performance Computing (HPC) platform in the University of Sheffield.

**model/best/**: The trained models are stored in this folder.

### B. speaker_verification_ghostnet

This folder includes implementations of the voiceprint verification in the cooperative module used in the proposed scheme.

**data_preprocess.py**: is used to process data.
**dataset.py**: defines the dataloader.
**ge2e_loss.py**: The loss function used to train the model.
**ghostnet.py**: The definition of Ghostnet model.
**hparam.py**: Functions used to process parameters defined in _config/config.yaml_.
**hpc.sh**: This is the HPC platform script.
**main.py**: is used to test the voiceprint verification submodule.
**train.py**: Train the Ghostnet.
**utils.py**: Help functions used in this module.
**config/config.yaml**: Defines the configration of this module.
**speech_id_checkpoint**: This folder is used to store the trained model.

### C. mobilefacenet_sge

This folder includes implementations of our system&#39;s non-cooperative authentication. 

**eval.py**: evaluates the result of the trained model.
**main.py**: test the face verification submodule.
**train.py**: is used to train the mobilefacenet model.

**core**: 
This folder contains the basic functions which are used by _train.py_. 
- **capturer.py**: captures and detects face using MTCNN from camera stream.
- **dataset.py**: defines dataloader.
- **evaluator.py**: contains the functions used for evaluate the model.
- **margin.py**: defines the [ArcFace loss](https://arxiv.org/abs/1801.07698).
- **model.py**: This is the definition of MobileFaceNet and MobileFaceNet with SGE.
- **utils.py**: includes the basic help functions used in this modules.

**HPC**: The bash files in this folder is the scripts used for submit jobs which are runned on the High Performance Computing (HPC) platform in the University of Sheffield.

**model/best/**: The trained models are stored in this folder.

### D. system

This part introduces the files which are runned on the robot, server and client respectively.

**robot**: This is the main file of the proposed system. All pretrained models are runned on the robot, the implementations of this part follows the flowchar of fig.2.

**server**: is account for training model, collect customer's registed data, etc. The robot can load related data from it.

**client**: receive PIN code from the server.

----
## **2. Guidance of simple experiment**

### Ⅰ. Formal Security Analysis in Cooperative Authentication

Put verifier.pv in folder that contains proverif.exe.

Open command (CMD).

Change the current working directory into the folder that has proverif.exe.

Insert the command &quot;_proverif verifier.pv_&quot; and press enter.

### Ⅱ. Non-cooperative authentication&#39;s simulation in Jupyter

Put the CUHK01 dataset under &quot;/data/&quot;, and run classify.py to add the training set and test set open any Jupyter file and run it directly. However, training and testing could be very time-consuming. We can also see the result of previous running directly in Jupyter files.

This part&#39;s work is briefly introduced in the video:

https://www.youtube.com/watch?v=zgD6tC4vGLM

## **3. Guidance of system implementation (demo).**

We will introduce the environmental requirements, and how to run the demo.

### Ⅰ. Environmental requirements:

#### **a. Cooperative off-line authentication:**

Table 1 shows the implementation environment of cooperative authentication. We can use two laptops as the client and the server, and Turtlebot3 as the robot. Here we install Ubuntu 16.04 in the client and the server and Ubuntu mate 16.04 in the robot. Then utilize ROS kinetic, which is recommended in Turtlebot3 and supports Python 2, in the server and the robot and set the server as the master.

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/tab1.png)

Table 1. Implementation environment of cooperative authentication

#### **b. Non-cooperative off-line authentication:**

The following table shows the implementation environment of non-cooperative authentication. We use the same devices of the cooperative part, but focus on the server and the robot. In addition, _virtualenv_ is utilized for building the Python 3 virtual environment in two devices, and we use Python 3 to fulfil requirement of some libraries in the area of computer vision. To train and test our model we utilize CUHK01 dataset.

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/tab2.png)

Table 2. Implementation environment of non-cooperative authentication

### Ⅱ. Detailed operation

#### **a. Cooperative off-line authentication demo:**

We can see the implementation of cooperative authentication in the video:

[_https://www.youtube.com/watch?v=-cANuZxD9uQ_](https://www.youtube.com/watch?v=-cANuZxD9uQ)

##### **Step C1** : Prepare work:

Set the environment based on Python 2 in Table 5.a.

Use ROS to create a map of the workplace.

(see https://emanual.robotis.com/docs/en/platform/turtlebot3/slam/)

Change the current working directory to &quot;/home/username/Desktop/project&quot; in three terminals.

Put true-client.py in the client.

Put true-robot.py in the robot.

Put true-server.py in the server.

##### **Step C2** : Launch ROS and robot equipment:

In the server, launch ROS: _roscore_

In the robot, launch the robot: _roslaunch turtlebot3\_bringup turtlebot3\_robot.launch_

##### **Step C3** : Run scripts:

Server launches a new terminal and start services: _python true-server.py_

Robot launches a new terminal and start services: _python true-robot.py_

Next, client sends the request: _python true-client.py_

As a result, the robot shows &quot;please input Y when robot arrives:&quot;, which means server has received the client&#39;s request and distributed information to the client and the robot.

##### **Step C4** : navigation:

Here we make a manual navigation.

Server launches a new terminal and insert:

_export TURTLEBOT3\_MODEL=waffle\_pi_

Then run the command to launch _Rviz_ for navigation:

_roslaunch turtlebot3\_navigation turtlebot3\_navigation.launch map\_file:=$HOME/true-map.yaml_

Press &quot;2D Pose Estimate&quot; to correct start position, and &quot;2D Nav Goal&quot; to select aimed destination. Robot plans the path and go there automatically as shown in Fig. 4.

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/fig4.png)

Figure 4. Navigation in the implementation of cooperative authentication

##### **Step C5** : QR code scan:

Upon arriving the destination, insert &quot;Y&quot; in the robot as a signal of completed navigation. Then, the robot automatically tries to scan QR code shown by the client. The scanning work succeeds and the QR code is authenticated, so the robot shows &quot;matched!&quot; and complete the delivery as shown in Fig. 5 and Fig. 6.

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/fig5.png)

Figure 5. QR code scanning in the implementation of cooperative authentication

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/fig6.png)

Figure 6. Result of QR code scanning in cooperative authentication

#### **b. Cooperative off-line authentication demo:**

This part executes cooperative authentication twice: A failed one before shifting to the non-cooperative part, and the other successful one when the robot recognizes the client and shifts the mode back.

We can see a video of the implementation above:

[_https://www.youtube.com/watch?v=lN5tngrdmds_](https://www.youtube.com/watch?v=lN5tngrdmds)

##### **Step N1:** Prepare work

Do prepare work in Step C1 in the implementation of cooperative authentication.

Set the virtual environment based on Python 3 in Table 5.b.

(see https://code-maven.com/slides/python/virtualev-python3)

Put capture.py and re\_identify\_robot.py in the robot.

Put classify.py, train.py, test\_CMC.py, re\_identify\_prepare.py in the server.

Put dataset (CUHK01) as &quot;/data/CUHK01/campus&quot;.

Next, server activates python3 environment and does pre-processing:

_source ~/venv3/bin/activate_

_python classify.py_

To finish the pre-processing work in the server, put images of the client into &quot;/data/reid\_prepare&quot;, and copy one of them into &quot;/data/reid\_robot&quot; in the robot as the comparison image. Therefore, we have four folders to store dataset:

/data/training\_set: Training set. Stored in the server.

/data/test\_set: Testing set. Stored in the server.

/data/reid\_prepare: All images of the client. Stored in the server.

/data/reid\_robot: An image of the client (a captured photo of pedestrian will be added here later).

##### **Step N2:** Model training

Then, train a model in the server:

_python train.py_

The model is stored as &quot;/net\_test.pth&quot;, and copy it to the robot.

##### **Step N3:** Optional Model testing

Run the code in the server to plot CMC in test1.jpg:

_python test\_CMC.py_

To get the separate line used in the final comparison, run the code in the server:

_python re\_identify\_prepare.py_

##### **Step N4:** Capturing images

Do the cooperative authentication and make the QR code scanning unit fail as shown in Fig. 7

Robots runs:

_python capture.py_

In this way, we successfully captured an image of pedestrian, did pedestrian detection in it, resize the person&#39;s image and stored it in &quot;/data/reid\_robot/&quot;.

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/fig7.png)

Figure 7. Failed QR code scanning and shifting to non-cooperative mode

##### **Step N5:** Person re-identification

Next, robot runs the command to compare the similarity of the captured image and client&#39;s image in &quot;/data/reid\_robot/&quot;.

_python re\_identify\_robot.py_

If the output is &quot;same person&quot;, robot successfully recognized the client, and the system switches back to cooperative mode as shown in Fig. 8 and Fig. 9.

##### **Step N6:** QR code scanning again

Finally, input any character in the robot&#39;s terminal that runs true-robot.py so it can start to scan the QR code again and match as shown in Fig. 10 and Fig. 11.

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/fig8.png)

Figure 8. Person detection and re-identification

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/fig9.png)

Figure 9. Result of person detection and re-identification

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/fig10.png)

Figure 10. QR code scanning again

![](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping/blob/master/Image/fig11.png)

Figure 11. Result of QR code scanning again

## **References:**
[Secure-Robotic-Shipping](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping)
[PyTorch_Speaker_Verification](https://github.com/HarryVolek/PyTorch_Speaker_Verification)
[GhostNet](https://github.com/huawei-noah/CV-Backbones)
[MobileFaceNet_Pytorch](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)














