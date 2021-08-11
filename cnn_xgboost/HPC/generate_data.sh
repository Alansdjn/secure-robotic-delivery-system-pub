#!/bin/bash
#$ -l h_rt=24:00:00  #time needed
#$ -pe smp 2 #number of cores
#$ -l rmem=6G #number of memery
#$ -o ./Output/genreate.log  #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M wwang107@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

#Load the conda module
module load apps/python/conda

#*Only needed if we're using GPU* Load the CUDA and cuDNN module
module load libs/cudnn/7.3.1.20/binary-cuda-9.0.176

#Load libsndfile
module load libs/libsndfile/1.0.28/gcc-4.9.4

#Activate the 'pytorch' environment
source activate pytorch

python ../generate_train_data.py
