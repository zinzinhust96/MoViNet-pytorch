import glob
import json
import os

import cv2
import numpy as np
from tqdm import tqdm


def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou


def get_expanded_box_with_ratio(cur_bbox, first_bbox, ratio, frame_shape):
    frame_height, frame_width, _ = frame_shape

    # calculate new bbox width and height
    expanded_height = first_bbox[3] - first_bbox[1]
    expanded_width = int(ratio * expanded_height)

    cur_bbox_center_x = (cur_bbox[0] + cur_bbox[2]) // 2
    cur_bbox_center_y = (cur_bbox[1] + cur_bbox[3]) // 2

    exp_x_min = max(cur_bbox_center_x - expanded_width // 2, 0)
    exp_y_min = max(cur_bbox_center_y - expanded_height // 2, 0)
    exp_x_max = min(cur_bbox_center_x + expanded_width // 2, frame_width)
    exp_y_max = min(cur_bbox_center_y + expanded_height // 2, frame_height)

    if exp_x_max == frame_width or exp_y_max == frame_height:
        exp_x_min = exp_x_max - expanded_width
        exp_y_min = exp_y_max - expanded_height
    else:
        exp_x_max = exp_x_min + expanded_width
        exp_y_max = exp_y_min + expanded_height

    return [exp_x_min, exp_y_min, exp_x_max, exp_y_max]

VIDEO_PATH = '/hdd/namdng/action_recognition/MoViNet-pytorch/scripts/clips'
DROP_PATH = '/hdd/namdng/action_recognition/MoViNet-pytorch/scripts/results'
BOX_RATIO = 1.5 # width/height = 1.5

if not os.path.exists(DROP_PATH):
    os.makedirs(DROP_PATH)

video_paths = sorted(glob.glob(os.path.join(VIDEO_PATH, 'raw', '*.mp4')))
# video_paths = ['/hdd/namdng/action_recognition/MoViNet-pytorch/scripts/clips_normal/demo_full_video_5s_before_theft0.mp4']
anno_paths = sorted(glob.glob(os.path.join(VIDEO_PATH, 'annotations', '*.json')))
# anno_paths = ['/hdd/namdng/action_recognition/MoViNet-pytorch/scripts/clips_normal/demo_full_video_5s_before_theft0.json']

for video_path, anno_path in tqdm(zip(video_paths, anno_paths), total=len(video_paths)):
    print(video_path, anno_path)
    base_name = os.path.basename(video_path)[:-4]
    with open(anno_path) as fopen:
        annotations = json.load(fopen)

    vid_capture = cv2.VideoCapture(video_path)
    if (vid_capture.isOpened() == False):
        assert "Error opening the video file"

    track_history = {}  # {track_id: {first_bbox: [], history: []}}
    frame_count = 0
    while(vid_capture.isOpened()):
        ret, frame = vid_capture.read()
        if not ret:
            break

        track_annos = annotations[frame_count]['track']
        person_tracks = [anno for anno in track_annos if anno['category'] == 0]
        motor_tracks = [anno for anno in track_annos if anno['category'] == 1]

        # tmp_history = {}
        for person_track in person_tracks:
            person_track_id = person_track['track_id']
            person_track_bbox = person_track['bbox']

            # check if this person overlap with any motor
            is_person_overlap = np.any([get_iou(person_track_bbox, motor_track['bbox']) > 0 for motor_track in motor_tracks])
            if is_person_overlap:
                if person_track_id in track_history:
                    # tmp_history[person_track_id] = track_history[person_track_id].copy()
                    first_bbox = track_history[person_track_id]['first_bbox']
                    expanded_bbox = get_expanded_box_with_ratio(person_track_bbox, first_bbox, BOX_RATIO, frame.shape)
                    track_history[person_track_id]['history'].append(frame[expanded_bbox[1]:expanded_bbox[3], expanded_bbox[0]:expanded_bbox[2]])
                else:
                    track_history[person_track_id] = {}
                    track_history[person_track_id]['first_bbox'] = person_track_bbox
                    expanded_bbox = get_expanded_box_with_ratio(person_track_bbox, person_track_bbox, BOX_RATIO, frame.shape)
                    track_history[person_track_id]['history'] = [frame[expanded_bbox[1]:expanded_bbox[3], expanded_bbox[0]:expanded_bbox[2]]]
            else:
                if person_track_id in track_history:
                    track_history.pop(person_track_id)

        # track_history = tmp_history.copy()

        frame_count += 1

    track_history_list = [(track_id, item['history']) for track_id, item in track_history.items()]
    track_history_list = sorted(track_history_list, key=lambda t: len(t[1]), reverse=True)
    for track_id, history in track_history_list:
        print(track_id, len(history))
        frame_height, frame_width, _ = history[0].shape
        if len(history) > 20 and frame_height > 100:
            out = cv2.VideoWriter(os.path.join(DROP_PATH, '{}_{}.mp4'.format(base_name, track_id)), cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
            for frame in history:
                out.write(frame)
    
