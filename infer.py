import argparse
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm

import transforms as T
from config import _C
from models import MoViNet


def extract_overlap_subclips(vid_capture, skip_frame, length_of_subclip = 16):
    frame_count = 0

    frame_list = []
    subclips = []
    while(vid_capture.isOpened()):
        ret, frame = vid_capture.read()
        if not ret:
            break

        if skip_frame == 0:
            frame_list.append(frame)
        elif frame_count % skip_frame == 0:
            frame_list.append(frame)
        if len(frame_list) == length_of_subclip:
            frame_array = np.stack(frame_list)
            subclips.append(frame_array)
            del frame_list[0]

        frame_count += 1

    return subclips

def run_test(model, data, args):
    model.eval()

    predictions = []
    model.clean_activation_buffers()
    with torch.no_grad():
        num_clips = data.shape[0]
        for i in tqdm(range(0, num_clips, args.bs_test)):
            # print('{} - {}'.format(i, min(i+args.bs_test, num_clips)))
            batch = data[i: min(i+args.bs_test, num_clips)]
            output = F.log_softmax(model(batch.to(args.device)), dim=1)
            _, pred = torch.max(output, dim=1)
            predictions.extend(pred.tolist())
            model.clean_activation_buffers()

    return predictions


def main(args):
    # Set cuda device
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    args.device = device

    transform_test = transforms.Compose([
                    T.ToFloatTensorInZeroOne(),
                    T.Resize((200, 200)),
                    #T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                    T.CenterCrop((172, 172))])

    # '''
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

    fps_output = 30
    skip_frame = math.ceil(fps/fps_output) - 1

    subclips = extract_overlap_subclips(vid_capture, skip_frame, args.num_frames)

    data = []
    for subclip_id, subclip in enumerate(subclips):
        video = torch.tensor(subclip)
        video = transform_test(video)
        data.append(torch.unsqueeze(video, 0))

    data = torch.cat(data, dim=0)
    # '''

    ''' by videoclips
    step_between_clips = 2
    frame_rate = 5
    video_clips = VideoClips(
        [args.video_path],
        args.num_frames,
        step_between_clips,
        frame_rate,
        num_workers=2
    )

    data = []
    for i in range(16):
        video, audio, _, video_idx = video_clips.get_clip(i)
        # print('before tranform: ', video.shape)
        video = transform_test(video)
        # print('after tranform: ', video.shape)
        data.append(torch.unsqueeze(video, 0))

    data = torch.cat(data, dim=0)
    # '''


    # initialize model
    model = MoViNet(_C.MODEL.MoViNetA0, 51,causal = False, pretrained = None, tf_like = True, device=device)
    model = model.to(device)
    if args.pretrained is not None and args.pretrained != 'None':
        model.load_state_dict(torch.load(args.pretrained, map_location=device))
        print("Restored from {}".format(args.pretrained))

    predictions = run_test(model, data, args)
    print('PREDICTION: ', predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Movinet inference')
    parser.add_argument("--video_path", required=True, help="path to video")
    parser.add_argument("--num_frames", type=int, default=16, help="number of frames in input")
    parser.add_argument("--bs_test", type=int, default=16, help="batch size test")
    parser.add_argument("--pretrained", default='/hdd/namdng/action_recognition/MoViNet-pytorch/results/theft/model_epoch0005_loss1.0017_acc77.00.pth', help="path to pre-trained model")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    args = parser.parse_args()
    print(args)
    main(args)
