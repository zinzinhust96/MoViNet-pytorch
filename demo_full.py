import json
import os
import time
from functools import wraps

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.multiprocessing import Queue

import transforms as T
from demo_cut import ActionRecognition
from utils import get_expanded_box_with_ratio, get_iou

current_milli_time = lambda: int(round(time.time() * 1000))

# try:
#     mp.set_start_method('spawn')
# except RuntimeError:
#     pass

VIDEO_FOLDER = '/hdd/namdng/action_recognition/MoViNet-pytorch/scripts/clips'
NAME = '0053_1'
VIDEO_PATH = os.path.join(VIDEO_FOLDER, 'raw', '{}.mp4'.format(NAME))
ANNO_PATH = os.path.join(VIDEO_FOLDER, 'annotations', '{}.json'.format(NAME))
BOX_RATIO = 1.5


class ImageGraber(mp.Process):
    def __init__(self, url, raw_frames):
        super(ImageGraber, self).__init__()
        # self.cam = cv2.VideoCapture(VIDEO_PATH)
        with open(ANNO_PATH) as fopen:
            annotations = json.load(fopen)
            
        self.annotations = annotations
        self.raw_frames = raw_frames

    def run(self):
        self.cam = cv2.VideoCapture(VIDEO_PATH)

        frame_count = 0
        while(self.cam.isOpened()):
            ret, frame = self.cam.read()
            # print('cam read: ', ret, frame.shape)
            if ret == False:
                time.sleep(0.001)
                continue

            self.raw_frames.put((frame, self.annotations[frame_count]['track']))
            # print('check empty: ', self.raw_frames.empty())
            if self.raw_frames.qsize() >= 5:
                self.raw_frames.get()

            frame_count += 1

class MainProcess(mp.Process):
    def __init__(self, raw_frames, pretrained, device):
        super(MainProcess, self).__init__()

        self.transform_test = transforms.Compose([
                        T.ToFloatTensorInZeroOne(),
                        T.Resize((200, 200))
                    ])

        ar_module = ActionRecognition(device)
        if pretrained is not None and pretrained != 'None':
            ar_module.load_model(pretrained)
        
        self.raw_frames = raw_frames
        self.ar_module = ar_module
        self.device = device
        self.track_history = {}  # {track_id: {first_bbox: [], history: [16 frames], last_expand_bbox: []}}


    def run(self):
        while True:
            if not self.raw_frames.empty():
                frame, track_annos = self.raw_frames.get()

                #
                person_tracks = [anno for anno in track_annos if anno['category'] == 0]
                motor_tracks = [anno for anno in track_annos if anno['category'] == 1]

                t0 = time.time()
                tmp_history = {}
                for person_track in person_tracks:
                    person_track_id = person_track['track_id']
                    person_track_bbox = person_track['bbox']

                    # check if this person overlap with any motor
                    is_person_overlap = np.any([get_iou(person_track_bbox, motor_track['bbox']) > 0 for motor_track in motor_tracks])
                    if is_person_overlap:
                        if person_track_id in self.track_history:
                            tmp_history[person_track_id] = self.track_history[person_track_id].copy()
                            first_bbox = tmp_history[person_track_id]['first_bbox']
                            expanded_bbox = get_expanded_box_with_ratio(person_track_bbox, first_bbox, BOX_RATIO, frame.shape)
                            if tmp_history[person_track_id]['history'].shape[0] == 16:
                                tmp_history[person_track_id]['history'] = tmp_history[person_track_id]['history'][1:]
                            tmp_history[person_track_id]['history'] = np.concatenate(
                                (
                                    tmp_history[person_track_id]['history'],
                                    np.array([frame[expanded_bbox[1]:expanded_bbox[3], expanded_bbox[0]:expanded_bbox[2]]]
                                )), axis=0)
                            tmp_history[person_track_id]['last_expand_bbox'] = expanded_bbox
                        else:
                            tmp_history[person_track_id] = {}
                            tmp_history[person_track_id]['first_bbox'] = person_track_bbox
                            expanded_bbox = get_expanded_box_with_ratio(person_track_bbox, person_track_bbox, BOX_RATIO, frame.shape)
                            tmp_history[person_track_id]['history'] = np.array([frame[expanded_bbox[1]:expanded_bbox[3], expanded_bbox[0]:expanded_bbox[2]]])
                            tmp_history[person_track_id]['last_expand_bbox'] = expanded_bbox
                    
                self.track_history = tmp_history.copy()

                #
                original_subclips = [(track_id, item['history'], item['last_expand_bbox']) for track_id, item in self.track_history.items() if item['history'].shape[0] == 16]
                print('cut frame time: ', time.time() - t0)
                if len(original_subclips) > 0:
                    t1 = time.time()
                    data = []
                    for track_id, subclip, _ in original_subclips:
                        # print('subclip: ', type(subclip))
                        # t2 = time.time()
                        video = torch.from_numpy(subclip)
                        print('video before transform: ', video.shape)
                        # print('t2: ', time.time() - t2)
                        # t3 = time.time()
                        video = self.transform_test(video)
                        # print('t3: ', time.time() - t3)
                        # t4 = time.time()
                        data.append(torch.unsqueeze(video, 0))
                        # print('t4: ', time.time() - t4)

                    data = torch.cat(data, dim=0)
                    print(data.shape, type(data))
                    print('transform time: ', time.time() - t1)

                    t2 = time.time()
                    predictions = self.ar_module.infer(data)
                    print('prediction: ', predictions)
                    print('infer time: ', time.time() - t2)
                    print('>>>>> total time: ', time.time() - t0)

                    # for idx, item in enumerate(original_subclips):
                    #     prediction = predictions[idx]
                    #     expand_bbox = item[2]
                    #     bbox_color = (0, 0, 255) if prediction == 1 else (0, 255, 0)  # LABEL 1 is theft
                    #     frame = cv2.rectangle(frame, (expand_bbox[0], expand_bbox[1]), (expand_bbox[2], expand_bbox[3]), bbox_color, thickness=2)
                        
                    #     cv2.imshow('demo', frame)
                    #     if cv2.waitKey(25) & 0xFF == ord('q'):
                    #         break

def main():
    gpu = 0
    device = 'cuda:{}'.format(gpu) if gpu != -1 else 'cpu'
    pretrained = '/hdd/namdng/action_recognition/MoViNet-pytorch/results/theft_2807_hw_200/model_epoch0010_loss0.6321_acc80.90.pth'

    raw_frames = Queue()    # TODO:

    grab_pr = ImageGraber(0, raw_frames)
    main_pr = MainProcess(raw_frames, pretrained, device)
    # disp_thr = ReceiveAndDisplay()
    grab_pr.start()
    main_pr.start()
    # disp_thr.start()

    grab_pr.join()
    main_pr.join()


if __name__ == "__main__":
    # mp.set_start_method('forkserver', force=True)
    # mp.set_start_method('forkserver')
    mp.set_start_method('spawn', force=True)
    main()
