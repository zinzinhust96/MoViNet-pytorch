import argparse
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import transforms as T
from config import _C
from models import MoViNet

torch.manual_seed(97)

def evaluate(model, data_load, args):
    model.eval()

    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0
    model.clean_activation_buffers()
    with torch.no_grad():
        for data, _, target in tqdm(data_load):
            output = F.log_softmax(model(data.to(args.device)), dim=1)
            loss = F.nll_loss(output, target.to(args.device), reduction='sum')
            _, pred = torch.max(output, dim=1)

            tloss += loss.item()
            csamp += pred.eq(target.to(args.device)).sum()
            model.clean_activation_buffers()
    aloss = tloss / samples
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')

    return aloss, 100.0 * csamp / samples


def main(args):
    # Set cuda device
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    args.device = device

    transform_test = transforms.Compose([
                                    T.ToFloatTensorInZeroOne(),
                                    T.Resize((200, 200)),
                                    #T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                    T.CenterCrop((172, 172))])

    # TODO: figure out the size of input
    print('processing test')
    hmdb51_test = torchvision.datasets.HMDB51(args.video_path,
                                              args.annotation_path,
                                              args.num_frames,
                                              frame_rate=5,
                                              step_between_clips = args.clip_steps,
                                              train=False,
                                              transform=transform_test,
                                              num_workers=2)

    test_loader  = DataLoader(hmdb51_test, batch_size=args.bs_test, shuffle=False)

    model = MoViNet(_C.MODEL.MoViNetA0, 51,causal = False, pretrained = None, tf_like = True, device=device)
    model = model.to(device)
    if args.pretrained is not None and args.pretrained != 'None':
        model.load_state_dict(torch.load(args.pretrained, map_location=device))
        print("Restored from {}".format(args.pretrained))

    val_loss, val_acc = evaluate(model, test_loader, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Movinet')
    parser.add_argument("--video_path", required=True, help="path to video")
    parser.add_argument("--annotation_path", required=True, help="path to validation data set")
    parser.add_argument("--num_frames", type=int, default=16, help="number of frames in input")
    parser.add_argument("--clip_steps", type=int, default=2, help="number of frames between subsclips")
    parser.add_argument("--bs_test", type=int, default=16, help="batch size test")
    parser.add_argument("--pretrained", default=None, help="path to pre-trained model")
    parser.add_argument("--gpu", type=int, default=1, help="gpu")
    args = parser.parse_args()
    print(args)
    main(args)
