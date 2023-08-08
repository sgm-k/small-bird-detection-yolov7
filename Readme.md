#  Winner Award: Small Object Detection Challenge for Spotting Birds
This repository contains the code for the method that secured the 1st place in the "Development Category" of the competition held in conjunction with MVA2023.

## Download the dataset with the following folder structure
Create a folder structure for the bird detection dataset as shown below:

bird_detection<br>
&emsp;│<br>
&emsp;└── dataset<br>
&emsp;&emsp;├── mva2023_sod4bird_pub_test<br>
&emsp;&emsp;└── mva2023_sod4bird_train<br>
## Convert the dataset
Run the script to prepare the dataset for training:

sh ./1_prepare_dataset.sh

## Training (Adjust parameters according to your environment)
Execute the script to start training the model. You may need to adjust the parameters depending on your system specifications.

sh ./2_train.sh

## Inference (Prepare the submission file)
Run the script to generate the submission file based on the trained model:

sh ./3_make_submit_file.sh
