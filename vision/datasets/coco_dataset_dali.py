from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import numpy as np
from time import time

class COCOPipeline(Pipeline):
    def __init__(self, file_root, annotations_file, batch_size, num_threads, device_id, num_gpus):
        super(COCOPipeline, self).__init__(batch_size, num_threads, device_id, seed = 15)
        self.input = ops.COCOReader(file_root = file_root,
                                    annotations_file = annotations_file,
                                    shard_id = device_id,
                                    num_shards = num_gpus,
                                    ratio=True,
                                    ltrb=True)
        self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
        self.flip = ops.Flip(device = "gpu")
        self.bbflip = ops.BbFlip(device = "cpu", ltrb=True)
        self.paste_pos = ops.Uniform(range=(0,1))
        self.paste_ratio = ops.Uniform(range=(1,2))
        self.coin = ops.CoinFlip(probability=0.5)
        self.coin2 = ops.CoinFlip(probability=0.5)
        self.paste = ops.Paste(device="gpu", fill_value=(32,64,128))
        self.bbpaste = ops.BBoxPaste(device="cpu", ltrb=True)
        self.prospective_crop = ops.RandomBBoxCrop(device="cpu",
                                                   aspect_ratio=[0.5, 2.0],
                                                   thresholds=[0.1, 0.3, 0.5],
                                                   scaling=[0.8, 1.0],
                                                   ltrb=True)
        self.slice = ops.Slice(device="gpu")

    def define_graph(self):
        rng = self.coin()
        rng2 = self.coin2()

        inputs, bboxes, labels = self.input()
        images = self.decode(inputs)

        # Paste and BBoxPaste need to use same scales and positions
        ratio = self.paste_ratio()
        px = self.paste_pos()
        py = self.paste_pos()
        images = self.paste(images, paste_x = px, paste_y = py, ratio = ratio)
        bboxes = self.bbpaste(bboxes, paste_x = px, paste_y = py, ratio = ratio)

        crop_begin, crop_size, bboxes, labels = self.prospective_crop(bboxes, labels)
        images = self.slice(images, crop_begin, crop_size)

        images = self.flip(images, horizontal = rng, vertical = rng2)
        bboxes = self.bbflip(bboxes, horizontal = rng, vertical = rng2)

        return (images, bboxes, labels)