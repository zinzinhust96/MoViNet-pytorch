import glob
import math
import os
import time

import cv2
import numpy as np
from tqdm import tqdm


def main(root_dir, out_dir, class_id, fps_output = 5, length_of_subclip = 16, ext = "mp4"):
    video_dir = os.path.join(root_dir, class_id)
    video_paths = glob.glob(os.path.join(video_dir, f"*.{ext}"))
    if out_dir is None:
        out_dir = root_dir

    for video_path in tqdm([video_paths[1]]):
        video_save_dir = os.path.join(out_dir, "clips_{}".format(class_id))     # clips_theft, clips_normal
        os.makedirs(video_save_dir, exist_ok=True)

        vid_capture = cv2.VideoCapture(video_path)
        if (vid_capture.isOpened() == False):
            assert "Error opening the video file"

        """ get video informations """
        fps = int(vid_capture.get(5))       # ~ vid_capture.get(cv2.CAP_PROP_FPS)
        frame_total = vid_capture.get(7)
        frame_width = int(vid_capture.get(3))
        frame_height = int(vid_capture.get(4))
        frame_size = (frame_width, frame_height)
        print(f'| Video info:\n - fps: {fps} \n - Frame number: {frame_total}, \n - Frame size: {frame_width}, {frame_height}')

        skip_frame = math.ceil(fps/fps_output) - 1

        subclips = extract_overlap_subclips(vid_capture, skip_frame, length_of_subclip)

        # save subclips
        file_name = os.path.basename(video_path)
        base_video_save_path = os.path.join(video_save_dir, file_name)
        for subclip_id, subclip in enumerate(subclips):
            video_save_path = base_video_save_path.replace(f".{ext}", "{:04d}.{}".format(subclip_id, ext))
            out = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_output, frame_size)
            for frame in subclip:
                out.write(frame)


def extract_overlap_subclips(vid_capture, skip_frame, length_of_subclip = 16):
    frame_count = 0

    frame_list = []
    subclips = []
    while(vid_capture.isOpened()):
        frame_count += 1
        ret, frame = vid_capture.read()
        if not ret:
            break

        if frame_count % skip_frame == 0:
            frame_list.append(frame)
        if len(frame_list) == length_of_subclip:
            frame_array = np.stack(frame_list)
            subclips.append(frame_array)
            del frame_list[0]

    return subclips

if __name__ == "__main__":
    root_dir = '/hdd/namdng/action_recognition/MoViNet-pytorch/data/video_data_1/'
    out_dir = "/mnt/smb106_motorbike/namdng_1"
    class_id = "brush_hair"
    ext = "avi"

    fps_output = 5              # fps of output video
    length_of_subclip = 16      # number frame per clip

    main(root_dir, out_dir, class_id, fps_output, length_of_subclip, ext)
