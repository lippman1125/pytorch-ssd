import torch
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite_xiaomi import create_mobilenetv2_ssd_lite_xiaomi, create_mobilenetv2_ssd_lite_predictor_xiaomi
from vision.ssd.fairnas_a_ssd_lite import create_fairnas_a_ssd_lite, create_fairnas_a_ssd_lite_predictor
from vision.ssd.fairnas_b_ssd_lite import create_fairnas_b_ssd_lite, create_fairnas_b_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.datasets.coco_dataset import CocoDatasetTest
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer
import argparse
import pathlib
import numpy as np
import logging
import sys
import os
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
import json


parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument("--trained_model", type=str)
parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument('--annfile', type=str, help='json annotation file, just for coco dataset')
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
parser.add_argument("--nms_method", type=str, default="hard")
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")


if __name__ == '__main__':
    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)
    timer = Timer()
    class_names = ["name"]*91
    dataset = CocoDatasetTest(args.dataset, args.annfile)

    if args.net == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(class_names), is_test=True)
    elif args.net == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True)
    elif args.net == 'mb2-ssd-lite-xiaomi':
        net = create_mobilenetv2_ssd_lite_xiaomi(len(class_names), width_mult=args.mb2_width_mult, is_test=True)
    elif args.net == 'fairnas-a-ssd-lite':
        net = create_fairnas_a_ssd_lite(len(class_names), is_test=True)
    elif args.net == 'fairnas-b-ssd-lite':
        net = create_fairnas_b_ssd_lite(len(class_names), is_test=True)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    timer.start("Load Model")
    net.load(args.trained_model)
    net = net.to(DEVICE)
    print('It took {} seconds to load the model.'.format(timer.end("Load Model")))
    if args.net == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd':
        predictor = create_mobilenetv1_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb1-ssd-lite':
        predictor = create_mobilenetv1_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'sq-ssd-lite':
        predictor = create_squeezenet_ssd_lite_predictor(net,nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb2-ssd-lite':
        predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'mb2-ssd-lite-xiaomi':
        predictor = create_mobilenetv2_ssd_lite_predictor_xiaomi(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'fairnas-a-ssd-lite':
        predictor = create_fairnas_a_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'fairnas-b-ssd-lite':
        predictor = create_fairnas_b_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Inference
    results = []
    for i in range(len(dataset)):
        print("process image", i)
        timer.start("Load Image")
        # get img & id
        image_id = dataset.ids[i]
        image = dataset.get_image(i)
        print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
        timer.start("Predict")
        boxes, labels, probs = predictor.predict(image)
        print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
        boxes = boxes.numpy()
        labels = labels.numpy()
        probs = probs.numpy()
        for k in range(np.shape(labels)[0]):
            result = {}
            result["image_id"] = int(image_id)
            result["category_id"] = int(labels[k])
            x1,y1,x2,y2 = boxes[k][0], boxes[k][1], boxes[k][2], boxes[k][3]
            result["bbox"] = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            result["score"] = float(round(probs[k], 2))

            results.append(result)

    # convert format from list to json
    results_json = json.dumps(results)

    with open(os.path.join(eval_path, "{}.json".format(args.net)), 'w') as f_ret:
        f_ret.write(results_json)







