from PIL import Image
import os
import os.path
import cv2
import numpy as np

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 8 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    return True

class CocoDataset:
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        # filter images without detection annotations
        ids = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                ids.append(img_id)
        self.ids = ids
        self.class_names = [
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
        '_',
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
        '_',
        'backpack',
        'umbrella',
        '_',
        '_',
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
        '_',
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
        '_',
        'dining table',
        '_',
        '_',
        'toilet',
        '_',
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
        '_',
        'book',
        'clock',
        'vase',
        'scissors',
        'teddy bear',
        'hair drier',
        'toothbrush' 
        ]
 
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        for ann in anns:
            # ann['bbox'][2:] += ann['bbox'][:2]
            x1,y1,w,h = ann['bbox'][0], ann['bbox'][1], ann['bbox'][2], ann['bbox'][3]
            # limit box size
            if w<=8 or h <=8:
                continue
            boxes.append(np.array([x1,y1,x1+w-1,y1+h-1], dtype=np.float32))
            labels.append(ann['category_id'])
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        # print(np.shape(boxes))
        # print(np.shape(labels))

        path = coco.loadImgs(img_id)[0]['file_name']

        image = self._read_image(os.path.join(self.root, path))
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        
        return image, boxes, labels


    def __len__(self):
        return len(self.ids)

    def _read_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_image(self, index):
        coco = self.coco
        img_id = self.ids[index]
        path = coco.loadImgs(img_id)[0]['file_name']
        image = self._read_image(os.path.join(self.root, path))
        if self.transform:
            image, _ = self.transform(image)
        return image

    # for eval
    def get_annotation(self, index):
        img_id = self.ids[index]
        return str(img_id), self._get_annotation(img_id)

    def _get_annotation(self, img_id):
        coco = self.coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            # ann['bbox'][2:] += ann['bbox'][:2]
            x1,y1,w,h = ann['bbox'][0], ann['bbox'][1], ann['bbox'][2], ann['bbox'][3]
            # limit box size
            if w<=1 or h <=1:
                continue
            boxes.append(np.array([x1,y1,x1+w-1,y1+h-1], dtype=np.float32))
            labels.append(ann['category_id'])
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        is_difficult = np.zeros(np.shape(labels)[0], dtype=np.uint8)

        return (boxes, labels, is_difficult)
