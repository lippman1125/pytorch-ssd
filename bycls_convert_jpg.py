import os
import types
import pickle
import random
import PIL
import collections
import numpy as np
import sys


input_dir = sys.argv[1]
labels_pkl = sys.argv[2]
output_dir = sys.argv[3]

if not os.path.exists(input_dir):
    print("invalid input_dir : {}".format(input_dir))
    exit(0)

if not os.path.exists(labels_pkl):
    print("invalid labels_pkl : {}".format(labels_pkl))
    exit(0)

if not os.path.exists(output_dir):
    print("makedirs={}".format(output_dir))
    os.makedirs(output_dir)

print("loading labels from pkl... " + labels_pkl)
with open(labels_pkl, "rb") as f:
    samples = pickle.load(f)

for s in samples:
    path, lable = s
    from PIL import Image
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # with open(os.path.join(input_dir, path), 'rb') as f:
    # img = Image.open(f).convert('RGB')
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    filename, suffix = os.path.splitext(basename)

    new_dir = os.path.join(output_dir, dirname)
    if not os.path.exists(new_dir):
        print("make new sub dir={}".format(new_dir))
        os.makedirs(new_dir)
    new_file = os.path.join(new_dir, filename + ".jpg")
    if not os.path.exists(new_file):
        print("old={} new={}".format(path, new_file))
        try:
            img = Image.open(os.path.join(input_dir, path)).convert("RGB")
            img.save(new_file)
        except(OSError, NameError):
            print("OSError : {}".format(path)) 
