import os
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

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

    if exp_x_max == 0 or exp_y_max == 0:
        exp_x_min = exp_x_max - expanded_width
        exp_y_min = exp_y_max - expanded_height
    else:
        exp_x_max = exp_x_min + expanded_width
        exp_y_max = exp_y_min + expanded_height

    return [exp_x_min, exp_y_min, exp_x_max, exp_y_max]
