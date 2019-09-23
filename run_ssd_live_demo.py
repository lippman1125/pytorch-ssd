from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite_xiaomi import create_mobilenetv2_ssd_lite_xiaomi, create_mobilenetv2_ssd_lite_predictor_xiaomi
from vision.ssd.fairnas_a_ssd_lite import create_fairnas_a_ssd_lite, create_fairnas_a_ssd_lite_predictor
from vision.ssd.fairnas_b_ssd_lite import create_fairnas_b_ssd_lite, create_fairnas_b_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if len(sys.argv) < 3:
    print('Usage: python run_ssd_example.py <net type>  <model path>  <data type> [video file]')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
data_type = sys.argv[3]
# label_path = sys.argv[3]

if len(sys.argv) >= 5:
    cap = cv2.VideoCapture(sys.argv[4])  # capture from file
else:
    cap = cv2.VideoCapture(0)   # capture from camera
    # cap.set(3, 960)
    # cap.set(4, 720)


# class_names = [name.strip() for name in open(label_path).readlines()]
class_names_voc = [
"BACKGROUND",
"aeroplane",
"bicycle",
"bird",
"boat",
"bottle",
"bus",
"car",
"cat",
"chair",
"cow",
"diningtable",
"dog",
"horse",
"motorbike",
"person",
"pottedplant",
"sheep",
"sofa",
"train",
"tvmonitor"
]

class_names_coco = [
'__background__',
'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
'kite', 'baseball bat', 'baseball glove', 'skateboard',
'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
'refrigerator', 'book', 'clock', 'vase', 'scissors',
'teddy bear', 'hair drier', 'toothbrush'
]

if data_type == "voc":
    num_classes = len(class_names_voc)
    class_names = class_names_voc
else:
    num_classes = len(class_names_coco)
    class_names = class_names_coco

if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(num_classes, is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(num_classes, is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(num_classes, is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(num_classes, is_test=True)
elif net_type == 'mb2-ssd-lite-xiaomi':
    net = create_mobilenetv2_ssd_lite_xiaomi(num_classes, is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(num_classes, is_test=True)
elif net_type == 'fairnas-a':
    net = create_fairnas_a_ssd_lite(num_classes, is_test=True)
elif net_type == 'fairnas-b':
    net = create_fairnas_b_ssd_lite(num_classes, is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)
net.to(DEVICE)

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=100, device=DEVICE)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=100, device=DEVICE)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=100, device=DEVICE)
elif net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=100, device=DEVICE)
elif net_type == 'mb2-ssd-lite-xiaomi':
    predictor = create_mobilenetv2_ssd_lite_predictor_xiaomi(net, candidate_size=100, device=DEVICE)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=100, device=DEVICE)
elif net_type == 'fairnas-a':
    predictor = create_fairnas_a_ssd_lite_predictor(net, candidate_size=100, device=DEVICE)
elif net_type == 'fairnas-b':
    predictor = create_fairnas_b_ssd_lite_predictor(net, candidate_size=100, device=DEVICE)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)


timer = Timer()
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, -1, 0.3)
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        # label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        label = "{}:{:.2f}".format(class_names[labels[i]], probs[i])
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        w, h = box[2] - box[0], box[3] -box[1]
        if w > 100 or h > 100:
            font_scale = 1
        elif w > 50 or h > 50:
            font_scale = 0.8
        elif w > 30 or h > 30:
            font_scale = 0.5
        else:
            font_scale = 0.3

        cv2.putText(orig_image, label,
                    (box[0], box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
