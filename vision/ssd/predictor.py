import torch

from ..utils import box_utils
from .data_preprocessing import PredictionTransform
from ..utils.misc import Timer
from ..utils.nms_gpu.nms import gpu_nms

class Predictor:
    def __init__(self, net, size, mean=0.0, std=1.0, nms_method=None,
                 iou_threshold=0.45, filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None, nms_gpu=False):
        self.net = net
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method
        self.nms_gpu = nms_gpu

        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net.to(self.device)
        self.net.eval()

        self.timer = Timer()

    def predict(self, image, top_k=-1, prob_threshold=None):
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            self.timer.start()
            scores, boxes = self.net.forward(images)
            print("Inference time: ", self.timer.end())
        boxes = boxes[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            if not self.nms_gpu:
                box_probs = box_utils.nms(box_probs, self.nms_method,
                                          score_threshold=prob_threshold,
                                          iou_threshold=self.iou_threshold,
                                          sigma=self.sigma,
                                          top_k=top_k,
                                          candidate_size=self.candidate_size)
            else:
                # convert tensor to numpy
                box_probs[:, 0] *= width
                box_probs[:, 1] *= height
                box_probs[:, 2] *= width
                box_probs[:, 3] *= height
                keep_id = gpu_nms(box_probs.numpy(), self.iou_threshold, device_id = 0)
                keep_id = keep_id[:self.candidate_size]
                box_probs = box_probs[keep_id, :]
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        if not self.nms_gpu:
            picked_box_probs[:, 0] *= width
            picked_box_probs[:, 1] *= height
            picked_box_probs[:, 2] *= width
            picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]
