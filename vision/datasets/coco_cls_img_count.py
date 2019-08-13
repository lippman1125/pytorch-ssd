from pycocotools.coco import COCO

class_names = [
        '__background__',
        'person',
        'bicycle',
        'car',
        'motorcycle',
        'airplane',
        'bus',
        'train',
        'truck',
        'boat',
        'traffic light',
        'fire hydrant',
        'stop sign',
        'parking meter',
        'bench',
        'bird',
        'cat',
        'dog',
        'horse',
        'sheep',
        'cow',
        'elephant',
        'bear',
        'zebra',
        'giraffe',
        'backpack',
        'umbrella',
        'handbag',
        'tie',
        'suitcase',
        'frisbee',
        'skis',
        'snowboard',
        'sports ball',
        'kite',
        'baseball bat',
        'baseball glove',
        'skateboard',
        'surfboard',
        'tennis racket',
        'bottle',
        'wine glass',
        'cup',
        'fork',
        'knife',
        'spoon',
        'bowl',
        'banana',
        'apple',
        'sandwich',
        'orange',
        'broccoli',
        'carrot',
        'hot dog',
        'pizza',
        'donut',
        'cake',
        'chair',
        'couch',
        'potted plant',
        'bed',
        'dining table',
        'toilet',
        'tv',
        'laptop',
        'mouse',
        'remote',
        'keyboard',
        'cell phone',
        'microwave',
        'oven',
        'toaster',
        'sink',
        'refrigerator',
        'book',
        'clock',
        'vase',
        'scissors',
        'teddy bear',
        'hair drier',
        'toothbrush' ]

annFile="/home/lqy/E/coco2014/annotations/instances_train2014.json"
coco = COCO(annFile)
print("@@@___coco train2014___")
coco_dict={}
for cls in class_names[1:]:
    catIds = coco.getCatIds(catNms=cls)
    imgIds = coco.getImgIds(catIds=catIds)
    print("class {}: {}".format(cls, len(imgIds)))
    coco_dict[cls] = len(imgIds)

annFile="/home/lqy/E/coco2014/annotations/instances_valminusminival2014.json"
coco = COCO(annFile)
print("@@@___coco val35k___")
for cls in class_names[1:]:
    catIds = coco.getCatIds(catNms=cls)
    imgIds = coco.getImgIds(catIds=catIds)
    print("class {}: {}".format(cls, len(imgIds)))
    coco_dict[cls] += len(imgIds)


print("@@@___coco trainval35k___")
for k,v in sorted(coco_dict.items(), key=lambda d:d[1]):
    print("class {}: {}".format(k, v))

