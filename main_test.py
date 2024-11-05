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

# import model
import torch.optim as optim
# from torch.optim import lr_scheduler
# from torchvision import transforms

import numpy as np
import random
import yaml
from tqdm import tqdm

import torch.nn.functional as F

from config_parser import *
from processor_test import Processor
if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(args).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()
    # init_seed(0)
    processor = Processor(arg)
    processor.start()