import glob
import os

import numpy as np
from sklearn.model_selection import train_test_split

VIDEO_PATH = '/hdd/namdng/action_recognition/MoViNet-pytorch/data/cut_clip_data_30fps/video_data/'
TYPES = ['theft', 'normal']
ANNO_PATH = '/hdd/namdng/action_recognition/MoViNet-pytorch/data/cut_clip_data_30fps/test_train_splits/'
TRAIN_TAG = 1
TEST_TAG = 2

for class_type in TYPES:
    video_paths = glob.glob(os.path.join(VIDEO_PATH, class_type, '*.mp4'))
    X_train, X_test = train_test_split(video_paths, test_size=0.2, random_state=42)

    # brush_hair_test_split1.txt
    with open(os.path.join(ANNO_PATH, '{}_test_split1.txt'.format(class_type)), 'w') as fwrite:
        for video_path in video_paths:
            video_name = os.path.basename(video_path)
            dataset_tag = TRAIN_TAG if video_path in X_train else TEST_TAG
            fwrite.write('{} {}\n'.format(video_name, dataset_tag))

