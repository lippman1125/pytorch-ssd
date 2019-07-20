import torch
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

class COCOPipeline(Pipeline):
    def __init__(self, default_boxes, root, annFile, batch_size,
                 local_rank, num_workers, seed):
        super(COCOPipeline, self).__init__(
            batch_size=batch_size,
            device_id=local_rank,
            num_threads=num_workers,
            seed=seed)


        # try:
        #     shard_id = torch.distributed.get_rank()
        #     num_shards = torch.distributed.get_world_size()
        # except RuntimeError:
        shard_id = 0
        num_shards = 1

        self.input = ops.COCOReader(
            file_root=root,
            annotations_file=annFile,
            skip_empty=True,
            shard_id=shard_id,
            num_shards=num_shards,
            ratio=True,
            ltrb=True,
            random_shuffle=False,
            shuffle_after_epoch=True)

        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)

        # Augumentation techniques
        self.crop = ops.RandomBBoxCrop(
            device="cpu",
            aspect_ratio=[0.5, 2.0],
            thresholds=[0, 0.1, 0.3, 0.5, 0.7, 0.9],
            scaling=[0.3, 1.0],
            ltrb=True,
            allow_no_crop=True,
            num_attempts=1)
        self.slice = ops.Slice(device="gpu")
        self.twist = ops.ColorTwist(device="gpu")
        self.resize = ops.Resize(
            device="gpu",
            resize_x=320,
            resize_y=320,
            min_filter=types.DALIInterpType.INTERP_TRIANGULAR)


        self.normalize = ops.CropMirrorNormalize(
            device="gpu",
            crop=(320, 320),
            mean=[127.0, 127.0, 127.0],
            std=[128.0, 128.0, 128.0],
            mirror=0,
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            pad_output=False)

        # Random variables
        self.rng1 = ops.Uniform(range=[0.5, 1.5])
        self.rng2 = ops.Uniform(range=[0.875, 1.125])
        self.rng3 = ops.Uniform(range=[-0.5, 0.5])

        self.flip = ops.Flip(device="gpu")
        self.bbflip = ops.BbFlip(device="cpu", ltrb=True)
        self.flip_coin = ops.CoinFlip(probability=0.5)

        self.box_encoder = ops.BoxEncoder(
            device="cpu",
            criteria=0.5,
            anchors=default_boxes.as_ltrb_list())


    def define_graph(self):
        saturation = self.rng1()
        contrast = self.rng1()
        brightness = self.rng2()
        hue = self.rng3()
        coin_rnd = self.flip_coin()

        inputs, bboxes, labels = self.input(name="Reader")
        images = self.decode(inputs)

        crop_begin, crop_size, bboxes, labels = self.crop(bboxes, labels)
        images = self.slice(images, crop_begin, crop_size)

        images = self.flip(images, horizontal=coin_rnd)
        bboxes = self.bbflip(bboxes, horizontal=coin_rnd)
        images = self.resize(images)
        images = images.gpu()
        images = self.twist(
            images,
            saturation=saturation,
            contrast=contrast,
            brightness=brightness,
            hue=hue)
        images = self.normalize(images)
        bboxes, labels = self.box_encoder(bboxes, labels)

        return (images, bboxes, labels)


def get_train_dali_loader(default_boxes, root, annFile, batch_size,
                 local_rank, num_workers, ngpus, local_seed):
    train_pipe = COCOPipeline(
        default_boxes,
        root,
        annFile,
        batch_size,
        local_rank,
        num_workers,
        seed=local_seed)

    train_loader = DALIGenericIterator(
        train_pipe,
        ["images", "boxes", "labels"],
        118287 / ngpus,
        stop_at_epoch=False)

    return train_loader
