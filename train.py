import argparse
import os
import shutil
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
from hmdb51_dataset import HMDB51
from models import MoViNet

torch.manual_seed(97)

def train_iter(model, optimz, data_load, args):
    samples = len(data_load.dataset)
    model.train()
    model.to(args.device)
    model.clean_activation_buffers()
    optimz.zero_grad()
    for i, (data,_ , target) in enumerate(tqdm(data_load)):
        out = F.log_softmax(model(data.to(args.device)), dim=1)
        loss = F.nll_loss(out, target.to(args.device))
        loss.backward()
        optimz.step()
        optimz.zero_grad()
        model.clean_activation_buffers()
        if i % args.log_interval == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))

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
    if not os.path.isdir(args.store_path):
        os.mkdir(args.store_path)

    # copy config to store path
    shutil.copy2('./train.sh', args.store_path)

    # Set cuda device
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    args.device = device

    transform = transforms.Compose([
                                    T.ToFloatTensorInZeroOne(),
                                    T.Resize((200, 200)),
                                    T.RandomHorizontalFlip(),
                                    #T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                    T.RandomCrop((172, 172))])
    transform_test = transforms.Compose([
                                    T.ToFloatTensorInZeroOne(),
                                    T.Resize((200, 200)),
                                    #T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                    T.CenterCrop((172, 172))])

    # input_size [16, 3, 16, 172, 172] (bs, num_channels, T, S, S)

    print('processing train')
    hmdb51_train = HMDB51(args.video_path,
                          args.annotation_path,
                          args.num_frames,
                          frame_rate=5,
                          step_between_clips = args.clip_steps,
                          train=True,
                          transform=transform,
                          num_workers=2)


    print('processing test')
    hmdb51_test = HMDB51(args.video_path,
                         args.annotation_path,
                         args.num_frames,
                         frame_rate=5,
                         step_between_clips = args.clip_steps,
                         train=False,
                         transform=transform_test,
                         num_workers=2)

    train_loader = DataLoader(hmdb51_train, batch_size=args.bs_train, shuffle=True)
    test_loader  = DataLoader(hmdb51_test, batch_size=args.bs_test, shuffle=False)

    model = MoViNet(_C.MODEL.MoViNetA0, 2,causal = False, pretrained = args.pretrained, tf_like = True, device=device)
    # model.classifier[3] = torch.nn.Conv3d(2048, 51, (1,1,1))
    optimz = optim.Adam(model.parameters(), lr=args.lr)
    # start_time = time.time()

    best_val_acc = 0
    for epoch in range(1, args.num_epochs + 1):
        print('Epoch:', epoch)
        train_iter(model, optimz, train_loader, args)
        val_loss, val_acc = evaluate(model, test_loader, args)
        if val_acc > best_val_acc:
            print('Saving best model ...')
            torch.save(model.state_dict(),
                    os.path.join(args.store_path, 'model_epoch{:04d}_loss{:.04f}_acc{:.02f}.pth'
                                .format(epoch, val_loss, val_acc)))
            best_val_acc = val_acc

# print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Movinet')
    parser.add_argument("--video_path", required=True, help="path to video")
    parser.add_argument("--annotation_path", required=True, help="path to validation data set")
    parser.add_argument("--num_frames", type=int, required=True, help="number of frames in input")
    parser.add_argument("--clip_steps", type=int, required=True, help="number of frames between subsclips")
    parser.add_argument("--bs_train", type=int, required=True, help="batch size train")
    parser.add_argument("--bs_test", type=int, required=True, help="batch size test")
    parser.add_argument("--lr", type=float, required=True, help="learning rate")
    parser.add_argument("--log_interval", type=int, default=100, help="train loss log interval")
    parser.add_argument("--num_epochs", default=50, help="number of epochs")
    parser.add_argument("--store_path", default='./results/debug', help="path to save trained model")
    parser.add_argument("--pretrained", default=None, help="path to pre-trained model")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    args = parser.parse_args()
    print(args)
    main(args)
