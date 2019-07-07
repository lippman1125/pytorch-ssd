import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
import sys

if len(sys.argv) != 3:
    print('Usage: python3 cocoeval.py  gt.json  dt.json')
    sys.exit(0)

annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print('Running demo for *%s* results.'%(annType))

#initialize COCO ground truth api
annFile = sys.argv[1]
cocoGt=COCO(annFile)

#initialize COCO detections api
resFile = sys.argv[2]
cocoDt=cocoGt.loadRes(resFile)

imgIds=sorted(cocoGt.getImgIds())

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()