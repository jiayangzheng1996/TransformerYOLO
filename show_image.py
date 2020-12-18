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
    num_class = len(VOC_CLASSES)
    num_grid = args.num_grid
    num_boxes = args.num_boxes
    hidden_dim = args.hidden_dim
    n_heads = args.n_heads
    num_encoder_layer = args.num_encoder_layer
    num_decoder_layer = args.num_decoder_layer

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    file_root_test = 'VOCdevkit_2007/VOC2007test/JPEGImages/'
    annotation_file_test = 'voc2007test.txt'

    test_dataset = VocDetectorDataset(root_img_dir=file_root_test, dataset_file=annotation_file_test, train=False,
                                      num_grid=num_grid)

    model = Detector(num_class, num_grid, num_boxes, args, hidden_dim, n_heads, num_encoder_layer, num_decoder_layer)
    model.to(device)

    model.load_state_dict(torch.load('best_model.pth'))

    # produce image output
    model.eval()
    for i in range(100):
        # select random image from test set
        image_name = random.choice(test_dataset.fnames)
        image = cv2.imread(os.path.join(file_root_test, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print('predicting...')
        result = predict_image(model, image_name, root_img_directory=file_root_test)
        for left_up, right_bottom, class_name, _, prob in result:
            color = COLORS[VOC_CLASSES.index(class_name)]
            cv2.rectangle(image, left_up, right_bottom, color, 2)
            label = class_name + str(round(prob, 2))
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            p1 = (left_up[0], left_up[1] - text_size[1])
            cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]),
                          color, -1)
            cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        cv2.imwrite("./output_images/output_%d.jpg" % i, image)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_grid', type=int, default=5)
    parser.add_argument('--num_boxes', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--num_encoder_layer', type=int, default=6)
    parser.add_argument('--num_decoder_layer', type=int, default=6)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--train_backbone', action='store_true')


    args = parser.parse_args()
    print(args)
    main(args)