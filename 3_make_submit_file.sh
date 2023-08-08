cp yolov7/runs/train/v7/weights/best.pt .
python3 submit.py
zip -r submit submit.json