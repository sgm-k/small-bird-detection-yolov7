import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from yolov7.utils.datasets import letterbox
import matplotlib.pyplot as plt

import glob
import json
import os
import numpy as np

# from PIL import Image

# import matplotlib
# matplotlib.use('TkAgg')

def flip_coords(coords, img_width):
    x1, y1, x2, y2 = coords
    new_x1 = img_width - x2
    new_x2 = img_width - x1
    return new_x1, y1, new_x2, y2

def convert_float32_to_float(data):
    if isinstance(data, list):
        return [convert_float32_to_float(x) for x in data]
    elif isinstance(data, dict):
        return {k: convert_float32_to_float(v) for k, v in data.items()}
    elif isinstance(data, np.float32):
        return float(data)
    else:
        return data
    
def iou(box1, box2):
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2

    intersection_x1 = max(x1a, x1b)
    intersection_y1 = max(y1a, y1b)
    intersection_x2 = min(x2a, x2b)
    intersection_y2 = min(y2a, y2b)

    intersection_area = max(intersection_x2 - intersection_x1, 0) * max(intersection_y2 - intersection_y1, 0)

    box1_area = (x2a - x1a) * (y2a - y1a)
    box2_area = (x2b - x1b) * (y2b - y1b)

    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area
import numpy as np
from ensemble_boxes import weighted_boxes_fusion,non_maximum_weighted

def nmw(all_dets, iou_thresh, conf_thresh):
    all_dets = np.array(all_dets)
    
    if len(all_dets) == 0:
        return []
    
    # Normalize coordinates
    width= 3840
    height = 2160
    normalized_dets = all_dets.copy()
    normalized_dets[:, 0] /= width
    normalized_dets[:, 1] /= height
    normalized_dets[:, 2] /= width
    normalized_dets[:, 3] /= height
    
    filtered_indices = np.where(all_dets[:, 4] >= conf_thresh)[0]
    normalized_dets = normalized_dets[filtered_indices]
    
    if len(normalized_dets) == 0:
        return []

    boxes = normalized_dets[:, :4]
    scores = normalized_dets[:, 4]
    labels = np.zeros_like(scores)

    filtered_boxes, filtered_scores, filtered_labels = weighted_boxes_fusion([boxes], [scores], [labels], iou_thr=iou_thresh)

    if len(filtered_boxes) == 0:
        return []

    filtered_dets = np.column_stack((filtered_boxes, filtered_scores))
    
    # Convert coordinates back to original image
    filtered_dets[:, 0] *= width
    filtered_dets[:, 1] *= height
    filtered_dets[:, 2] *= width
    filtered_dets[:, 3] *= height

    return filtered_dets
# from ensemble_boxes import weighted_boxes_fusion
def wbf(all_dets, iou_thresh, conf_thresh):
    all_dets = np.array(all_dets)

    if len(all_dets) == 0:
        return []

    filtered_indices = np.where(all_dets[:, 4] >= conf_thresh)[0]
    all_dets = all_dets[filtered_indices]

    if len(all_dets) == 0:
        return []

    sorted_indices = np.argsort(all_dets[:, 4])[::-1]
    filtered_dets = []

    while len(sorted_indices) > 0:
        current_det = all_dets[sorted_indices[0]]
        filtered_dets.append(current_det)

        other_dets = all_dets[sorted_indices[1:]]
        ious = np.array([iou(current_det[:4], det[:4]) for det in other_dets])

        weights = other_dets[:, 4] * (ious >= iou_thresh)
        current_det[:4] = (current_det[:4] * current_det[4] + np.sum(other_dets[:, :4] * weights[:, None], axis=0)) / (current_det[4] + np.sum(weights))

        sorted_indices = sorted_indices[1:]
        sorted_indices = sorted_indices[ious < iou_thresh]

    return filtered_dets
def nms(all_dets, iou_thresh, conf_thresh):
    all_dets = np.array(all_dets)
    
    if len(all_dets) == 0:
        return []
    
    filtered_indices = np.where(all_dets[:, 4] >= conf_thresh)[0]
    all_dets = all_dets[filtered_indices]

    if len(all_dets) == 0:
        return []

    sorted_indices = np.argsort(all_dets[:, 4])[::-1]
    filtered_dets = []

    while len(sorted_indices) > 0:
        current_det = all_dets[sorted_indices[0]]
        filtered_dets.append(current_det)

        other_dets = all_dets[sorted_indices[1:]]
        ious = np.array([iou(current_det[:4], det[:4]) for det in other_dets])

        sorted_indices = sorted_indices[1:]
        sorted_indices = sorted_indices[ious < iou_thresh]

    return filtered_dets

def soft_nms(all_dets, iou_thresh, conf_thresh, sigma=0.5):
    all_dets = np.array(all_dets)
    
    if len(all_dets) == 0:
        return []
    
    filtered_indices = np.where(all_dets[:, 4] >= conf_thresh)[0]
    all_dets = all_dets[filtered_indices]

    if len(all_dets) == 0:
        return []

    sorted_indices = np.argsort(all_dets[:, 4])[::-1]
    filtered_dets = []

    while len(sorted_indices) > 0:
        current_index = sorted_indices[0]
        current_det = all_dets[current_index]
        filtered_dets.append(current_det)

        other_indices = sorted_indices[1:]
        other_dets = all_dets[other_indices]
        ious = np.array([iou(current_det[:4], det[:4]) for det in other_dets])

        # Update the confidence scores based on the IoUs and sigma
        conf_updates = np.exp(-(ious ** 2) / sigma)
        all_dets[other_indices, 4] *= conf_updates

        # Remove the processed detection and apply the confidence threshold
        sorted_indices = np.delete(sorted_indices, 0)
        high_conf_indices = np.where(all_dets[sorted_indices, 4] >= conf_thresh)
        sorted_indices = sorted_indices[high_conf_indices]

    return filtered_dets

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    # weights= "./best-v7.pt"
    weights= "./best.pt"

    # tile_size = (1920, 1080)
    tile_size = (2880, 1620)

    # overlap = (1440, 810)
    # overlap = (480, 270)
    overlap = (960, 540)
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # img_paths=glob.glob("../dataset/mva2023_sod4bird_pub_test/images/*")
    img_paths=glob.glob("../../bird_yolov7/bird/dataset/mva2023_sod4bird_pub_test/images/*")
    json_file = []
    for source in img_paths:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

        original_image = cv2.imread(source)
        original_height, original_width = original_image.shape[:2]

        y_steps = ((original_height - tile_size[1]) // overlap[1]) + 1
        x_steps = ((original_width - tile_size[0]) // overlap[0]) + 1

        all_dets = []

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1


        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Flip image
            img_flipped = torch.flip(img, dims=[3])  # Flip horizontally

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # Inference
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]
                pred_flipped = model(img_flipped, augment=opt.augment)[0]

            # Non-maximum suppression
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            pred_flipped = non_max_suppression(pred_flipped, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

            # Process detections
            for i, det in enumerate(pred):
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for *xyxy, conf, cls in det.detach().cpu().numpy():
                        x1, y1, x2, y2 = xyxy
                        all_dets.append([x1, y1, x2, y2, conf, cls])

            for i, det in enumerate(pred_flipped):
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Convert flipped coordinates back to the original image
                    det[:, [0, 2]] = im0.shape[1] - det[:, [2, 0]]

                    for *xyxy, conf, cls in det.detach().cpu().numpy():
                        x1, y1, x2, y2 = xyxy
                        all_dets.append([x1, y1, x2, y2, conf, cls])
                        
        dataset = LoadImages(source, img_size=3360, stride=stride)
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # Inference
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]

            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            for i, det in enumerate(pred):
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for *xyxy, conf, cls in det.detach().cpu().numpy():
                        x1, y1, x2, y2 = xyxy
                        # print(x1, y1, x2, y2, conf, cls)
                        all_dets.append([x1, y1, x2, y2, conf, cls])


        filtered_dets = nmw(all_dets, opt.iou_thres, opt.conf_thres)
        dataset = LoadImages(source, img_size=3520, stride=stride)
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # Inference
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=opt.augment)[0]

            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            for i, det in enumerate(pred):
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for *xyxy, conf, cls in det.detach().cpu().numpy():
                        x1, y1, x2, y2 = xyxy
                        # print(x1, y1, x2, y2, conf, cls)
                        all_dets.append([x1, y1, x2, y2, conf, cls])


        filtered_dets = nmw(all_dets, opt.iou_thres, opt.conf_thres)
        # width= 3840
        # height = 2160
        # filtered_dets[:, 0] *= width
        # filtered_dets[:, 1] *= height
        # filtered_dets[:, 2] *= width
        # filtered_dets[:, 3] *= height
        # filtered_dets = nms(all_dets, opt.iou_thres, opt.conf_thres)
        # filtered_dets = nms(filtered_dets, 0.9, opt.conf_thres)
        # print(filtered_dets)

        for det in filtered_dets:
            if len(det) == 6:
                x1, y1, x2, y2, conf, cls = det
            elif len(det) == 5:
                x1, y1, x2, y2, conf = det
                cls = 0  # デフォルトのクラス値
            result={}

            basename = os.path.basename(source)
            base = basename.split(".")[0]

            result = {
                "image_id": int(base),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(conf),
                "category_id": int(cls)
            }
            print(result["image_id"],result["score"],result["bbox"])

            json_file.append(result)

        # Draw bounding boxes on the original image
        # for det in filtered_dets:
        #     x1, y1, x2, y2, conf, cls = det
        #     label = f'{int(cls)} {conf:.2f}'
        #     plot_one_box([x1, y1, x2, y2], original_image, label=label, color=[random.randint(0, 255) for _ in range(3)])

        max_width = 1500
        max_height = 1500

        # Calculate the new dimensions while preserving the aspect ratio
        height, width = original_image.shape[:2]
        new_width = max_width
        new_height = int(height * (max_width / float(width)))

        if new_height > max_height:
            new_width = int(width * (max_height / float(height)))
            new_height = max_height

        # Resize the image while preserving the aspect ratio
        resized_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        # json_file = convert_float32_to_float(json_file)

        # with open('submit.json', 'a') as jfile:
        #     json.dump(json_file, jfile)
        # Display the resized image with bounding boxes
        # cv2.imshow('Resized Image with Bounding Boxes', resized_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    json_file = convert_float32_to_float(json_file)       
    jfile = open('submit.json', 'w')
    json.dump(json_file, jfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=3200, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
