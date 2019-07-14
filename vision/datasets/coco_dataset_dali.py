from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import numpy as np
from time import time

class COCOPipeline(Pipeline):
    def __init__(self, tfrecord_path, tfrecord_idx_path, batch_size, num_threads, device_id, target_transform=None):
        super(COCOPipeline, self).__init__(batch_size, num_threads, device_id, seed = 15)
        self.input = ops.TFRecordReader(path = tfrecord_path,
                                        index_path = tfrecord_idx_path,
                                        features = {
                                        'image/encoded' :            tfrec.FixedLenFeature((), tfrec.string, ""),
                                        'image/height':              tfrec.FixedLenFeature((), tfrec.int64, 1),
                                        'image/width':               tfrec.FixedLenFeature((), tfrec.int64, 1),
                                        'image/filename':            tfrec.FixedLenFeature((), tfrec.string, ''),
                                        'image/source_id':           tfrec.FixedLenFeature((), tfrec.string, ''),
                                        'image/key/sha256':          tfrec.FixedLenFeature((), tfrec.string, ''),
                                        'image/format':              tfrec.FixedLenFeature((), tfrec.string, 'jpeg'),
                                        'image/object/bbox/xmin':    tfrec.VarLenFeature(tfrec.float32, 0.0),
                                        'image/object/bbox/ymin':    tfrec.VarLenFeature(tfrec.float32, 0.0),
                                        'image/object/bbox/xmax':    tfrec.VarLenFeature(tfrec.float32, 0.0),
                                        'image/object/bbox/ymax':    tfrec.VarLenFeature(tfrec.float32, 0.0),
                                        'image/object/class/label':  tfrec.VarLenFeature(tfrec.int64, -1),
                                        'image/object/class/text':   tfrec.VarLenFeature(tfrec.string, ''),
                                        'image/object/is_crowd':     tfrec.VarLenFeature(tfrec.int64,  0),
                                        'image/object/area':         tfrec.VarLenFeature(tfrec.float32, 0.0),
                                        })
        self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
        self.flip = ops.Flip(device = "gpu")
        self.bbflip = ops.BbFlip(device = "cpu", ltrb=True)
        self.paste_pos = ops.Uniform(range=(0,1))
        self.paste_ratio = ops.Uniform(range=(1,2))
        self.coin = ops.CoinFlip(probability=0.5)
        self.paste = ops.Paste(device="gpu", fill_value=(32,64,128))
        self.bbpaste = ops.BBoxPaste(device="cpu", ltrb=True)
        self.prospective_crop = ops.RandomBBoxCrop(device="cpu",
                                                   aspect_ratio=[0.5, 2.0],
                                                   thresholds=[0.1, 0.3, 0.5, 0.7, 0.9],
                                                   scaling=[0.8, 1.0],
                                                   ltrb=True)

        self.resize = ops.Resize(device = "gpu", resize_x = 320.0, resize_y = 320.0)
        self.normalize = ops.NormalizePermute(device="gpu",
                                              mean=[127, 127, 127],
                                              std=[128.0],
                                              image_type=types.RGB,
                                              output_type=types.FLOAT)
        self.slice = ops.Slice(device="gpu")
        self.target_transform = target_transform

    def define_graph(self):
        inputs = self.input()
        images = self.decode(inputs['image/encoded'])
        labels = self.inputs['image/object/class/label']
        xmin = self.input['image/object/bbox/xmin']
        ymin = self.input['image/object/bbox/ymin']
        xmax = self.input['image/object/bbox/xmax']
        ymax = self.input['image/object/bbox/ymax']
        bboxes = np.hstack(xmin, ymin, xmax, ymax)

        rng = self.coin()

        # Paste and BBoxPaste need to use same scales and positions
        ratio = self.paste_ratio()
        px = self.paste_pos()
        py = self.paste_pos()
        images = self.paste(images, paste_x = px, paste_y = py, ratio = ratio)
        bboxes = self.bbpaste(bboxes, paste_x = px, paste_y = py, ratio = ratio)

        # Random Crop
        crop_begin, crop_size, bboxes, labels = self.prospective_crop(bboxes, labels)
        images = self.slice(images, crop_begin, crop_size)

        # Random Flip
        images = self.flip(images, horizontal = rng)
        bboxes = self.bbflip(bboxes, horizontal = rng)
        # Resize
        images = self.resize(images)
        # Normalize
        images = self.normalize(images)
        if self.target_transform:
            boxes, labels = self.target_transform(bboxes, labels)

        return (images, bboxes, labels)