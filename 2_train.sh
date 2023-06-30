cd yolov7
pip3 install -r requirements.txt
python3 train.py --workers 8 --device 0 --batch-size 1 --data data/bird.yaml --img 3200 3200 --cfg cfg/training/yolov7.yaml --weights yolov7_training.pt --name v7 --hyp data/origin.yaml
cd ..
#python3 train.py --workers 8 --device 0,1,2,3 --batch-size 3 --data data/bird.yaml --img 3200 3200 --cfg cfg/training/yolov7.yaml --weights runs/train/v7_bird_3200_data-14/weights/epoch_699.pt --name v7_bird_3200_data- --hyp data/origin.yaml