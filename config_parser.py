import sys, os, argparse, time

import argparse
import inspect
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict
from torchinfo import summary
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
# import datasets
from feeders import dataset as dataset
from feeders import utils
import torch.optim as optim
# from torch.optim import lr_scheduler
# from torchvision import transforms

import numpy as np
import random
import yaml
from tqdm import tqdm

import torch.nn.functional as F

def get_parser():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head Pose Estimation network.')

    parser.add_argument(
        '--work_dir',
        default='./work_dir/temp',
        help='the work folder for storing results')
    parser.add_argument('--model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/train_config.yaml',
        help='path to the configuration file test_21_landmark')

    # processor
    # parser.add_argument(
    #     '--phase', default='train', help='must be train or test')
    # parser.add_argument(
    #     '--save-score',
    #     type=str2bool,
    #     default=False,
    #     help='if ture, the classification score will be stored')

    # dataset
    parser.add_argument('--train_dataset',
        help='Dataset type.',
        default='Pose_300W_LP', type=str)
    parser.add_argument('--train_data_path',
        help='Directory path for data.',
        default='data/SynergyNet/300W_LP/', type=str)
    parser.add_argument('--train_file_name',
        help='Path to text file containing relative paths for every example.',
        default='filelist/300W_LP-3D.txt', type=str)
    
    parser.add_argument('--valid_dataset',
        help='Dataset type.',
        default='Pose_300W_LP', type=str)
    parser.add_argument('--valid_data_path',
        help='Directory path for data.',
        default='data/SynergyNet/300W_LP/', type=str)
    parser.add_argument('--valids_file_name',
        help='Path to text file containing relative paths for every example.',
        default='filelist/300W_LP-3D.txt', type=str)

    parser.add_argument('--test_dataset1',
        help='Dataset type.',
        default='Pose_300W_LP', type=str)
    parser.add_argument('--test_data_path1',
        help='Directory path for data.',
        default='data/SynergyNet/300W_LP/', type=str)
    parser.add_argument('--test_file_name1',
        help='Path to text file containing relative paths for every example.',
        default='filelist/300W_LP-3D.txt', type=str)

    parser.add_argument('--test_dataset2',
        help='Dataset type.',
        default='Pose_300W_LP', type=str)
    parser.add_argument('--test_data_path2',
        help='Directory path for data.',
        default='data/SynergyNet/300W_LP/', type=str)
    parser.add_argument('--test_file_name2',
        help='Path to text file containing relative paths for every example.',
        default='filelist/300W_LP-3D.txt', type=str)

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 50, 80],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')

    #=============================================================================
    parser.add_argument(
        '--batch-size', type=int, default=128, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=128, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=100,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--debug', default=False, help='debug or not debug')
    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    parser.add_argument('--num_worker', default=8)

    parser.add_argument('--continue_training', default=False, help='continue training or not')
    parser.add_argument('--weights_file_extract', default=False, help='continue training or not')
    parser.add_argument('--checkpoint_file', default='', help='load checkpoint file')
    parser.add_argument('--test_loop', default=False, help='test all model')
    parser.add_argument('--weights_file', default='', help='model file')


    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=2,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')
    
    parser.add_argument(
        '--weight2', default=None, help='Weight for model 2')

    return parser

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False