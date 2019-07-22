import os
import types
import pickle
import random
import PIL
import collections
import numpy as np
import sys
import glob

input_dir = sys.argv[1]
output_dir = sys.argv[2]

if not os.path.exists(input_dir):
    print("invalid input_dir : {}".format(input_dir))
    exit(0)

if not os.path.exists(output_dir):
    print("makedirs={}".format(output_dir))
    os.makedirs(output_dir)

samples = glob.glob(os.path.join(input_dir, "*.*"))

for s in samples:
    path = s
    from PIL import Image
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # with open(os.path.join(input_dir, path), 'rb') as f:
    # img = Image.open(f).convert('RGB')
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    filename, suffix = os.path.splitext(basename)

    new_file = os.path.join(output_dir, filename + ".jpg")
    if not os.path.exists(new_file):
        print("old={} new={}".format(path, new_file))
        try:
            img = Image.open(os.path.join(path)).convert("RGB")
            img.save(new_file)
        except(OSError, NameError):
            print("OSError : {}".format(path)) 
