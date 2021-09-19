# **Handbook**

## **Project Title:**

**An AI-driven Secure and Intelligent Robotic Delivery System**

## **Please see Code files in GitHub** :

[https://github.com/Alansdjn/secure-robotic-delivery-system-pub](https://github.com/Alansdjn/secure-robotic-delivery-system-pub)

## **Brief Introduction:**

**This document is divided into three parts:**

1. Explanation of files in project _An AI-driven Secure and Intelligent Robotic Delivery System_.

2. Guidance of system implementation.

**Datasets:**

- [Speech Commands](https://arxiv.org/abs/1804.03209): As the PIN code is consists of digit numbers only, we only selected a subset of the whole commands, i.e., 0, 1, 2 up to 9 from the speech commands data set.

- [TIMIT](https://github.com/philipperemy/timit)

- [CASIA-WebFace](https://arxiv.org/pdf/1411.7923v1.pdf)

- [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/)

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

This folder includes implementations of the voiceprint verification in the cooperative module which are used in the proposed scheme.

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

## **2. Guidance of system implementation.**

We will introduce the environmental requirements, and how to run the demo.

### Ⅰ. Environmental requirements:

Table 1 shows the implementation environment of the proposed scheme. We use two laptops as the client and the server, and Turtlebot3 as the robot. Here, a Macbook with MacOS 11.04 is used as a client, we install Ubuntu 18.04 on the server and Ubuntu mate 18.04 on the robot. Ubuntu 18.04 can be download from the [TurtleBot offical website](https://emanual.robotis.com/docs/en/platform/turtlebot3/sbc_setup/). The ROS we choose melodic, althouth we do not use ROS in out project. This is because the bounded OS supports the 3rd party libraries used in this project. All implementions in the experiment were developed using python based on the pytorch framework.  _virtualenv_ is utilized for building the Python 3 virtual environment in three devices, and the Python version is 3.7. Training and testing were carried out on the High Performance Computing (HPC) platform in the University of Sheffield, running on NVIDIA K80 GPU. The utilized datasets are listed in previous section. 

![](https://github.com/Alansdjn/secure-robotic-delivery-system-pub/blob/main/images/tab1.png)

Table 1. Implementation environment

### Ⅱ. Detailed operation

#### **System experimental demo:**

We have recorded an experimental demo and upload it to the [youtube](https://youtu.be/1yWgYfRGoVs). Now, let's introduce how to build up the system.

##### **Step 1** : Prepare work:

Set the environment based on Python 3 in Table 1.

As I met some changes when installing librosa in Ubuntu 18.04 on the robot. So I list the method here:

    pip install soundfile
    
    sudo apt-get update
    
    sudo apt-get install libsndfile1-dev gfortran libatlas-base-dev libopenblas-dev liblapack-dev -y

    pip install sndfile

    # Building scipy from source takes about 1hr20. Pre-compiled wheels are available 
    # from piwheels.org, so you can install it from there without building yourself.
    pip install scipy --extra-index-url https://www.piwheels.org/simple

    pip install scikit-learn --extra-index-url https://www.piwheels.org/simple
    
    #sudo apt-get install llvm
    # llvm-6.0
    # LLVM 6.0-41~exp5~ubuntu1, ==> llvmlite 0.23.0~0.26.0 => numba 0.36.2
    #LLVM_CONFIG=/usr/bin/llvm-config pip install llvmlite==0.26.0 numba==0.41.0

    #First, install llvm-9
    sudo apt install llvm-9-dev
    
    #Then, relink 'llvm-config'
    sudo ln -s /usr/bin/llvm-config-9 /usr/bin/llvm-config
    LLVM_CONFIG=/usr/bin/llvm-config pip install llvmlite==0.33.0 numba==0.41.0

    pip install librosa

##### **Step 2** : Generate training data:

Since we use HPC to train and test our model, so we should upload the dataset to the platform. And then process the training data by submitting `./cnn_xgboost/HPC/generate_data.sh` or running script `data_preprocess.py` directly on HPC platform.

##### **Step 3** : Train and test model:

As the training and tesing datasets are generated, we can submit the jobs to train and test the networks.
For the CNN-XGBoost hybrid model, we train the CNN first using the bash file `./cnn_xgboost/HPC/train_cnn3.sh`, and then fine tune the XGBoost classifer by submit the job `./cnn_xgboost/HPC/cnn_xgb3_tune.sh`. For the voiceprint identification model, it can be trained using `./speaker_verification_ghostnet/hpc.sh`. We use `./mobilefacenet_sge/hpc.sh` to train the face identification model. All trained models are stored under the folder `model/best`.

##### **Step 4** : Run the system:

Load the Python code to the robot, client, and server, and then run the `robot.py`, `client.py` and `server.py` script respectively. The server should be runned first, because both robot and client need to connect to the server and request messages from it. Figure 3 shows a screen shoot of the experiment.

![](https://github.com/Alansdjn/secure-robotic-delivery-system-pub/blob/main/images/fig3.png)

Figure 3. run the system

##### **Step 5** : PIN code verification:

When the authentication process is triggered, the robot will run the PIN code verification. To complete the PIN code verification, the process can be further divided into 6 steps which are shown in Figure 4. The first 3 steps are used to collect and pre-process data. The robot records audio and splits it into segments. Then, convert them to Mel spectrograms. The followed 2 steps are in charge of extracting features using a CNN network, and classifying the features to 10 digit classes using XGBoost classifier. The last step joins the digits together, and check if it is correct.

![](https://github.com/Alansdjn/secure-robotic-delivery-system-pub/blob/main/images/fig3.png)

Figure 4. Steps of PIN code verification

##### **Step 6** : Voiceprint verification:

To complete the voiceprint verification, we can re-use the recorded audio in the previous step. Unlike the previous module, we do not need to split the audio. Here, we use a lightweight network Ghostnet as the feature extractor. It is specially designed for end device. Last, the cosine similarity of the extracted features and the registered features is calculated to determine whether this is the correct customer.

![](https://github.com/Alansdjn/secure-robotic-delivery-system-pub/blob/main/images/fig5.png)

Figure 5. PIN code verification and voiceprint verification 


##### **Step 7** : Face verification:

To finish the face verification, there are 4 major steps: 
	1. capture a frame from the video stream, 
	2. detect face using MTCNN network, 
	3. extract face features via improved MobileFaceNet, and then 
	4. calculate the cosine similarity to verify the identification.

In this section, we proposed an improved MobileFaceNet to extract features.

![](https://github.com/Alansdjn/secure-robotic-delivery-system-pub/blob/main/images/fig5.png)

Figure 6. Face verification

## **References:**
1. [Secure-Robotic-Shipping](https://github.com/Jiapei-Yang/Secure-Robotic-Shipping)
2. [PyTorch_Speaker_Verification](https://github.com/HarryVolek/PyTorch_Speaker_Verification)
3. [GhostNet](https://github.com/huawei-noah/CV-Backbones)
4. [MobileFaceNet_Pytorch](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)
5. J. Yang, P. Gope, Y. Cheng, and L. Sun, “Design, analysis and implementation of a smart next generation secure shipping infrastructure using autonomous robot,” Computer Networks, vol. 187, p. 107779, 2021.
6. K. Han, Y. Wang, Q. Tian, J. Guo, C. Xu, and C. Xu, “Ghostnet: More features from cheap operations,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 1580–1589.
7. S. Chen, Y. Liu, X. Gao, and Z. Han, “Mobilefacenets: Efficient cnns for accurate real-time face verification on mobile devices,” in Chinese Conference on Biometric Recognition. Springer, 2018, pp. 428–438.
8. X. Li, X. Hu, and J. Yang, “Spatial group-wise enhance: Improving semantic feature learning in convolutional networks,” arXiv preprint arXiv:1905.09646, 2019.
9. P. Warden, “Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition,” ArXiv e-prints, Apr. 2018. [Online]. Available: https://arxiv.org/abs/1804.03209 
10. J. S. Garofolo, “Timit acoustic phonetic continuous speech corpus,” Linguistic Data Consortium, 1993, 1993. 
11. D. Yi, Z. Lei, S. Liao, and S. Z. Li, “Learning face representation from scratch,” arXiv preprint arXiv:1411.7923, 2014. 
12. G. B. Huang, M. Mattar, T. Berg, and E. Learned-Miller, “Labeled faces in the wild: A database forstudying face recognition in unconstrained environments,” in Workshop on faces in’Real-Life’Images: detection, alignment, and recognition, 2008.
13. H. Y. Khdier, W. M. Jasim, and S. A. Aliesawi, “Deep learning algorithms based voiceprint recognition system in noisy environment,” in Journal of Physics: Conference Series, vol. 1804, no. 1. IOP Publishing, 2021, p. 012042.














