import os
import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import random

import cv2

from dataset.dataset import VocDetectorDataset
from dataset.config import VOC_CLASSES, COLORS
from model.transformer_detector import Detector
from model.loss import DetectorLoss
from model.predict import predict_image

from engine import train, eval, evaluate

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    num_class = len(VOC_CLASSES)
    num_epoch = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    lr_backbone = args.lr_backbone
    weight_decay = args.weight_decay
    num_grid = args.num_grid
    num_boxes = args.num_boxes
    l_coord = args.coord
    l_noobj = args.noobj
    hidden_dim = args.hidden_dim
    n_heads = args.n_heads
    num_encoder_layer = args.num_encoder_layer
    num_decoder_layer = args.num_decoder_layer

    writer = SummaryWriter()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    file_root_train = 'VOCdevkit_2007/VOC2007/JPEGImages/'
    annotation_file_train = 'voc2007.txt'

    train_dataset = VocDetectorDataset(root_img_dir=file_root_train, dataset_file=annotation_file_train, train=True,
                                       num_grid=num_grid)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    print('Loaded %d train images' % len(train_dataset))

    file_root_test = 'VOCdevkit_2007/VOC2007test/JPEGImages/'
    annotation_file_test = 'voc2007test.txt'

    test_dataset = VocDetectorDataset(root_img_dir=file_root_test, dataset_file=annotation_file_test, train=False,
                                      num_grid=num_grid)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    print('Loaded %d test images' % len(test_dataset))

    model = Detector(num_class, num_grid, num_boxes, args, hidden_dim, n_heads, num_encoder_layer, num_decoder_layer)
    model.to(device)

    param_dict = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
         "lr": lr_backbone
         }
    ]
    optimizer = optim.AdamW(param_dict, lr=lr, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    criterion = DetectorLoss(num_grid, num_boxes, l_coord, l_noobj, device=device)

    training_loss = []
    testing_loss = []
    eval_map = []

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    for epoch in range(num_epoch):
        train(epoch, model, criterion, optimizer, train_loader, device, args, writer, training_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            eval(epoch, model, criterion, test_loader, annotation_file_test, device, args, writer, testing_loss, eval_map)
        lr_scheduler.step()
        print()
        print()
        print()

    # loss plots
    plt.figure()
    x_training = np.arange(1, 51, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.plot(x_training, training_loss)
    plt.show()

    x_testing = []
    for i in range(51):
        if i == 1 or (i % 5 == 0 and i != 0):
            x_testing.append(i)
    plt.xlabel("Epoch")
    plt.ylabel("Testing Loss")
    plt.plot(x_testing, testing_loss)
    plt.show()

    plt.xlabel("Epoch")
    plt.ylabel("Testing mAP")
    plt.plot(x_testing, eval_map)
    plt.show()

    writer.flush()
    writer.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--num_grid', type=int, default=5)
    parser.add_argument('--num_boxes', type=int, default=2)
    parser.add_argument('--coord', type=float, default=5, help='YOLO coefficient for regression loss')
    parser.add_argument('--noobj', type=float, default=0.5, help='YOLO coefficient for no object confidence loss')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--num_encoder_layer', type=int, default=6)
    parser.add_argument('--num_decoder_layer', type=int, default=6)
    parser.add_argument('--train_backbone', action='store_true')
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--max_norm', type=float, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_every', type=int, default=10)


    args = parser.parse_args()
    print(args)
    main(args)