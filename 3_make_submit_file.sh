cd yolov7
cp runs/train/v7/weights/best.pt .
python3 submit.py
zip -r submit submit.json