cd coco2yolo
python3 drone-coco2yolo.py -ann-path ../dataset/mva2023_sod4bird_train/annotations/split_train_coco.json -img-dir ../dataset/mva2023_sod4bird_train/images -task-dir ../dataset/train
python3 drone-coco2yolo.py -ann-path ../dataset/mva2023_sod4bird_train/annotations/split_val_coco.json -img-dir ../dataset/mva2023_sod4bird_train/images -task-dir ../dataset/valid
cd ../