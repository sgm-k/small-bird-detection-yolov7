# Download the dataset with the following folder structure
Create a folder structure for the bird detection dataset as shown below:

bird_detection
│
└── dataset
    ├── mva2023_sod4bird_pub_test
    └── mva2023_sod4bird_train
# Convert the dataset
Run the script to prepare the dataset for training:

sh ./1_prepare_dataset.sh

# Training (adjust parameters according to your environment)
Execute the script to start training the model. You may need to adjust the parameters depending on your system specifications.

sh ./2_train.sh

# Inference (prepare the submission file)
Run the script to generate the submission file based on the trained model:

sh ./3_make_submit_file.sh