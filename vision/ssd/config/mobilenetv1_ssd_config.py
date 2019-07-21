import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

# for VOC training
# image_size = 300
# for COCO training
image_size = 320
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.6
center_variance = 0.1
size_variance = 0.2

# specs = [
#     SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
#     SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
#     SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
#     SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
#     SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
#     SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
# ]
# specs = [
#     # feature size : stride : box size : box ratio
#     SSDSpec(19, 16, SSDBoxSizes(30, 60), [2, 3]),
#     SSDSpec(10, 32, SSDBoxSizes(60, 111), [2, 3]),
#     SSDSpec(5, 64, SSDBoxSizes(111, 162), [2, 3]),
#     SSDSpec(3, 100, SSDBoxSizes(162, 213), [2, 3]),
#     SSDSpec(2, 150, SSDBoxSizes(213, 264), [2, 3]),
#     SSDSpec(1, 300, SSDBoxSizes(264, 315), [2, 3])
#]
specs = [
    # feature size : stride : box size : box ratio
    SSDSpec(20, 16, SSDBoxSizes(21, 45), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(45, 99), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(99, 153), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(153, 207), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(207, 261), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(261, 315), [2, 3])
]


priors = generate_ssd_priors(specs, image_size)
