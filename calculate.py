import json
from PIL import Image
import os

# 打开并读取 JSON 文件
def readimg(filename):
    with Image.open(filename+".png") as img:
        width, height = img.size

    with open(filename+".json", 'r', encoding='utf-8') as file:
        data = json.load(file)
        boxes = []
        for _, box in data.items():
            for b in box:
                boxes.append([b['column_min']/width, b['row_min']/height,b['column_max']/width,b['row_max']/height])
    return boxes


import csv
def readcsv(filename):
    with open(filename, newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=' ')

        boxes = []
        for row in reader:
            boxes.append(list(map(lambda x: float(x), row[1:])))
    return boxes


def yolo_to_bbox(yolo_box):
    x_center, y_center, width, height = yolo_box
    x_min = (x_center - width / 2)
    y_min = (y_center - height / 2)
    x_max = (x_center + width / 2) 
    y_max = (y_center + height / 2)
    return [x_min, y_min, x_max, y_max]


def calculate_iou(box1, box2):
    # 计算交集的坐标
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # 计算交集的面积
    width_inter = max(0, x2_inter - x1_inter)
    height_inter = max(0, y2_inter - y1_inter)
    area_inter = width_inter * height_inter

    # 计算每个矩形的面积
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集的面积
    area_union = area_box1 + area_box2 - area_inter

    # 计算 IoU
    iou = area_inter / area_union
    return iou




def get_filenames(directory):
    """
    获取指定目录下的所有文件名（不包括路径）。

    :param directory: 目录的路径
    :return: 文件名列表
    """
    filenames = []
    files = os.listdir(directory)
    files.sort()  # 按文件名排序

    for filename in files:
        if os.path.isfile(os.path.join(directory, filename)):
            filenames.append(filename)

    return filenames



yololine = []
xianyuline = []


for iouthreshlod in range(10,90,10):
    baselinebox = []
    yolobox = []
    files = []
    for file in get_filenames("test/labels"):
        files.append(file)
        box1 = readcsv(f"test/labels/{file}")
        box2 = readcsv(f"test/yolo/{file}")
        baselinebox.append(box1)
        yolobox.append(box2)

    xianyuboxes = []
    validation_files = []
    for file in get_filenames("test/images"):
        if not file.endswith("json"):
            continue
        validation_files.append(file)

        box = readimg("/mnt/hardDisk1/fay/datasets/webelement-3/test/images/"+file.rstrip('.json'))
        xianyuboxes.append(box)

    xianyucnt = 0
    yolocnt = 0
    boxtotal = 0
    for index, box in enumerate(baselinebox):
        xianyu = xianyuboxes[index]
        yolo = yolobox[index]

        print(f"filename {files[index]} ===============")
        print(f"filename {validation_files[index]} ===============")

        boxtotal += len(box)    

        for v in xianyu:
            for b in box:
                iou = calculate_iou(yolo_to_bbox(b), v)
                if iou != 0 and iou > iouthreshlod/100:
                    xianyucnt += 1
                    
        for v in yolo:
            for b in box:
                iou = calculate_iou(yolo_to_bbox(b), yolo_to_bbox(v))
                if iou != 0 and iou > iouthreshlod/100:
                    yolocnt += 1
        print("\n")



    xianyuline.append(xianyucnt/boxtotal)
    yololine.append(yolocnt/boxtotal)
print(f"xianyu precision = {xianyuline}")
print(f"yolo precision = {yololine}")

import matplotlib.pyplot as plt

# 假设这是您的数据
iou_thresholds = [i/100 for i in range(10,90,10)]  # IoU 阈值

plt.plot(iou_thresholds, xianyuline, marker='o', label='Traditional')  # 'o' 表示圆形点标记
plt.plot(iou_thresholds, yololine, marker='s', label='Deep Learning')      # 's' 表示正方形点标记

plt.xlabel('IoU Threshold')
plt.ylabel('Score')
plt.title('Recall')
plt.grid(True)
plt.legend()  # 显示图例
plt.savefig('line.png')