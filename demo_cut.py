import argparse
import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm

import transforms as T
from config import _C
from models import MoViNet


class ActionRecognition():
    def __init__(self, device):
        super().__init__()
        model = MoViNet(_C.MODEL.MoViNetA0, 2,causal = False, pretrained = None, tf_like = True, device=device)
        model = model.to(device)

        self.device = device
        self.model = model

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print("Restored from {}".format(model_path))

    def infer(self, data):
        self.model.eval()

        self.model.clean_activation_buffers()
        with torch.no_grad():
            output = F.log_softmax(self.model(data.to(self.device)), dim=1)
            _, pred = torch.max(output, dim=1)
            self.model.clean_activation_buffers()

        return pred


def main(args):
    # Set cuda device
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    args.device = device

    transform_test = transforms.Compose([
                        T.ToFloatTensorInZeroOne(),
                        T.Resize((200, 200))
                    ])

    """ initialize model """
    ar_module = ActionRecognition(device)
    if args.pretrained is not None and args.pretrained != 'None':
        ar_module.load_model(args.pretrained)

    vid_capture = cv2.VideoCapture(args.video_path)
    if (vid_capture.isOpened() == False):
        assert "Error opening the video file"

    """ get video informations """
    fps = int(vid_capture.get(5))       # ~ vid_capture.get(cv2.CAP_PROP_FPS)
    frame_total = vid_capture.get(7)
    frame_width = int(vid_capture.get(3))
    frame_height = int(vid_capture.get(4))
    frame_size = (frame_width, frame_height)
    print(f'| Video info:\n - fps: {fps} \n - Frame number: {frame_total}, \n - Frame size: {frame_width}, {frame_height} \n - Frame total: {frame_total}')

    INPUT_FPS = 10
    skip_frame = math.ceil(fps/INPUT_FPS) - 1

    frame_count = 0

    frame_list = []
    prediction = None
    video_out = cv2.VideoWriter(os.path.join(args.output_video, os.path.basename(args.video_path)), cv2.VideoWriter_fourcc(*'mp4v'), INPUT_FPS, (frame_width, frame_height+40))
    while(vid_capture.isOpened()):
        ret, frame = vid_capture.read()
        if not ret:
            break

        frame_array = None
        if skip_frame == 0:
            frame_list.append(frame)
        elif frame_count % skip_frame == 0:
            frame_list.append(frame)
        if len(frame_list) == args.num_frames:
            frame_array = np.stack(frame_list)
            del frame_list[0]

        if frame_array is not None:
            video = torch.tensor(frame_array)
            video = transform_test(video)   # (num_channels, T, S, S)
            input_data = torch.unsqueeze(video, 0)  # (1, num_channels, T, S, S)

            prediction = ar_module.infer(input_data)[0]

        # output demo frame
        border_color = (0, 0, 255) if prediction == 1 else (0, 255, 0)  # LABEL 1 is theft
        padded_frame = cv2.copyMakeBorder(frame, 0, 40, 0, 0, cv2.BORDER_CONSTANT)
        padded_frame = cv2.rectangle(padded_frame, (0, 0), (frame_width, frame_height), border_color, thickness=2)
        video_out.write(padded_frame)
        # cv2.imshow('demo', padded_frame)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break

        frame_count += 1
    
    video_out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Movinet inference')
    parser.add_argument("--num_frames", type=int, default=16, help="number of frames in input")
    parser.add_argument("--bs_test", type=int, default=16, help="batch size test")
    parser.add_argument("--video_path", default='/hdd/namdng/action_recognition/MoViNet-pytorch/scripts/demo_videos/0041_1_110.mp4', help="path to video")
    parser.add_argument("--pretrained", default='/hdd/namdng/action_recognition/MoViNet-pytorch/results/theft_2807_hw_200/model_epoch0010_loss0.6321_acc80.90.pth', help="path to pre-trained model")
    parser.add_argument("--output_video", default='/hdd/namdng/action_recognition/MoViNet-pytorch/results/demo/', help="path to demo output video folder")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    args = parser.parse_args()
    print(args)
    main(args)
