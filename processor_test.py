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
from model.ref_dis_noise_old_nll_v2 import Model 
from config_parser import *

import torch.optim as optim
import tools
import numpy as np
import random
import yaml
from tqdm import tqdm
import torch.nn.functional as F

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)




class Processor():
    """
        Processor for Head Pose Estimation
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        # if arg.phase == 'train':
        # #     if not arg.train_feeder_args['debug']:
        #     if not arg.debug:
        #         if os.path.isdir(arg.model_saved_name):
        #             print('log_dir: ', arg.model_saved_name, 'already exist')
        #             answer = input('delete it? y/n:')
        #             if answer == 'y':
        #                 shutil.rmtree(arg.model_saved_name)
        #                 print('Dir removed: ', arg.model_saved_name)
        #                 input('Refresh the website of tensorboard by pressing any keys')
        #             else:
        #                 print('Dir not removed: ', arg.model_saved_name)
        #         self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train_tensorboard'), 'train')
        #         self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val_tensorboard'), 'val')
        #     else:
        #         self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test_tensorboard'), 'test')
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0

        if self.arg.continue_training:
            load_cp = self.load_checkpoint(self.model, self.optimizer, self.arg.checkpoint_file)
            self.model = load_cp[0]
            self.optimizer = load_cp[1]

    def load_data(self):
        # Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()

        if self.arg.train_dataset == 'Pose_300W_LP':
            train_dataset = dataset.Pose_300W_LP(self.arg.train_data_path, self.arg.train_file_name, self.arg.train_data_path, self.arg.train_file_name)
        elif self.arg.train_dataset == 'AFLW2000':
            train_dataset = dataset.AFLW2000(self.arg.train_data_path, self.arg.train_file_name, self.arg.train_data_path, self.arg.train_file_name)
        elif self.arg.train_dataset == 'BIWI':
            train_dataset = dataset.BIWI(self.arg.train_data_path, self.arg.train_file_name, self.arg.train_data_path, self.arg.train_file_name)
        else:
            print('Error: not a valid dataset name')
            sys.exit()

        if self.arg.test_dataset1 == 'Pose_300W_LP':
            test_dataset1 = dataset.Pose_300W_LP(self.arg.test_data_path1, self.arg.test_file_name1, self.arg.train_data_path, self.arg.train_file_name)
        elif self.arg.test_dataset1 == 'AFLW2000':
            test_dataset1 = dataset.AFLW2000(self.arg.test_data_path1, self.arg.test_file_name1, self.arg.train_data_path, self.arg.train_file_name)
        elif self.arg.test_dataset1 == 'BIWI':
            test_dataset1 = dataset.BIWI(self.arg.test_data_path1, self.arg.test_file_name1, self.arg.train_data_path, self.arg.train_file_name)
        else:
            print('Error: not a valid dataset name')
            sys.exit()

        if self.arg.test_dataset2 == 'Pose_300W_LP':
            test_dataset2 = dataset.Pose_300W_LP(self.arg.test_data_path2, self.arg.test_file_name2, self.arg.train_data_path, self.arg.train_file_name)
        elif self.arg.test_dataset2 == 'AFLW2000':
            test_dataset2 = dataset.AFLW2000(self.arg.test_data_path2, self.arg.test_file_name2, self.arg.train_data_path, self.arg.train_file_name)
        elif self.arg.test_dataset2 == 'BIWI':
            test_dataset2 = dataset.BIWI(self.arg.test_data_path2, self.arg.test_file_name2, self.arg.train_data_path, self.arg.train_file_name)
        else:
            print('Error: not a valid dataset name')
            sys.exit()

        self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)

        self.data_loader['test1'] = torch.utils.data.DataLoader(
            dataset=test_dataset1,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

        self.data_loader['test2'] = torch.utils.data.DataLoader(
            dataset=test_dataset2,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        # Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        print(self.model)

        # Loss function
        self.softmax = nn.Softmax().cuda(output_device)
        self.criterion = nn.CrossEntropyLoss().cuda(output_device)
        self.reg_criterion = nn.MSELoss().cuda(output_device)

        # if self.arg.weights:
        #     self.global_step = int(arg.weights[:-3].split('-')[-1])
        #     self.print_log('Load weights from {}.'.format(self.arg.weights))
        #     if '.pkl' in self.arg.weights:
        #         with open(self.arg.weights, 'r') as f:
        #             weights = pickle.load(f)
        #     else:
        #         weights = torch.load(self.arg.weights)

        #     weights = OrderedDict(
        #         [[k.split('module.')[-1],
        #           v.cuda(output_device)] for k, v in weights.items()])

        #     keys = list(weights.keys())
        #     for w in self.arg.ignore_weights:
        #         for key in keys:
        #             if w in key:
        #                 if weights.pop(key, None) is not None:
        #                     self.print_log('Sucessfully Remove Weights: {}.'.format(key))
        #                 else:
        #                     self.print_log('Can Not Remove Weights: {}.'.format(key))

        #     try:
        #         self.model.load_state_dict(weights)
        #     except:
        #         state = self.model.state_dict()
        #         diff = list(set(state.keys()).difference(set(weights.keys())))
        #         print('Can not find these weights:')
        #         for d in diff:
        #             print('  ' + d)
        #         state.update(weights)
        #         self.model.load_state_dict(state)

        # if type(self.arg.device) is list:
        #     if len(self.arg.device) > 1:
        #         self.model = nn.DataParallel(
        #             self.model,
        #             device_ids=self.arg.device,
        #             output_device=output_device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        lr_scheduler_pre = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.arg.step, gamma=0.5)

        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, total_epoch=self.arg.warm_up_epoch,
                                                   after_scheduler=lr_scheduler_pre)
        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def checkpoint(self):
        state_dict = self.model.state_dict()
        weights = OrderedDict([[k.split('module.')[-1],
                                v.cpu()] for k, v in state_dict.items()])
        checkpoint = {
            "epoch": 100,
            "model_state": weights,
            "optim_state": self.optimizer.state_dict()
        }

        # checkpoint_1 = {
        #     "epoch": 100,
        #     "model_state": state_dict,
        #     "optim_state": self.optimizer.state_dict()
        # }

        torch.save(checkpoint, self.arg.model_saved_name + "-" + "checkpoint.pt")
        # torch.save(checkpoint_1, self.arg.model_saved_name + "-" + "checkpoint_1.pt")

    # def load_checkpoint(self, model, optimizer, cp_file):
    #     # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    #     start_epoch = 0
    #     if os.path.isfile(cp_file):
    #         print("=> loading checkpoint '{}'".format(cp_file))
    #         checkpoint = torch.load(cp_file)
    #         start_epoch = checkpoint['epoch']
    #         # model.load_state_dict(checkpoint['model_state'])
    #         weights = OrderedDict(
    #             [[k.split('module.')[-1],
    #               v.cuda(self.output_device)] for k, v in checkpoint['model_state'].items()])
    #         model.load_state_dict(weights)
    #         optimizer.load_state_dict(checkpoint['optim_state'])
    #         # losslogger = checkpoint['losslogger']
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(cp_file, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(cp_file))

    #     return model, optimizer, start_epoch

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()


        # speed up
        cudnn.fastest = True
        cudnn.benchmark = True
        cudnn.deterministic = False
        cudnn.enabled = True

        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        if not self.arg.continue_training:
            self.adjust_learning_rate(epoch)
        # for name, param in self.model.named_parameters():
        #     self.train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        loss_value = []
        # self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
     
        self.yaw_error = 0
        self.roll_error = 0
        self.pitch_error = 0
        total = 0
        for batch_idx, (img, ref, matrix_label, euler_label, ref_matrix_label, ref_euler_label, index) in enumerate(process):
            # get data
            total += img.size(0)
            img = Variable(img.float().cuda(self.output_device), requires_grad=False)
            euler_label = Variable(euler_label.float().cuda(self.output_device), requires_grad=False)
            matrix_label = Variable(matrix_label.float().cuda(self.output_device), requires_grad=False)
            
            ref = Variable(ref.float().cuda(self.output_device), requires_grad=False)
            ref_euler_label = Variable(ref_euler_label.float().cuda(self.output_device), requires_grad=False)
            ref_matrix_label = Variable(ref_matrix_label.float().cuda(self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()
         
            label_yaw = euler_label[:, 0]
            label_pitch = euler_label[:, 1]
            label_roll = euler_label[:, 2]

            # forward
            pre_matrix = self.model(img, ref, ref_matrix_label)
          
            # pose_loss = self.reg_criterion(pre_matrix, matrix_label)
            pose_loss = matrix_loss(pre_matrix, matrix_label)    
            # Total loss
            
            loss = pose_loss
            
    
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            # gradient clip
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()
            pre_matrix = torch.clamp(pre_matrix,min=-1,max=1)
            pre_matrix = tools.symmetric_orthogonalization(pre_matrix)
            pre_pitch, pre_yaw, pre_roll = utils.mat_to_euler(pre_matrix.view(-1,3,3).detach())
            pre_yaw = torch.rad2deg(pre_yaw).cpu()
            pre_pitch = torch.rad2deg(pre_pitch).cpu()
            pre_roll = torch.rad2deg(pre_roll).cpu()
            self.yaw_error += torch.sum(torch.abs(pre_yaw - label_yaw.cpu()))
            self.pitch_error += torch.sum(torch.abs(pre_pitch - label_pitch.cpu()))
            self.roll_error += torch.sum(torch.abs(pre_roll - label_roll.cpu()))
         
            
            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
          
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log(
            '\tlearning rate: {:.4f}.'.format(self.optimizer.param_groups[0]['lr']))
        self.print_log(
            '\tyaw error: {:.4f}.'.format(self.yaw_error/total))
        self.print_log(
            '\tpitch error: {:.4f}.'.format(self.pitch_error/total))
        self.print_log(
            '\troll error: {:.4f}.'.format(self.roll_error/total))

        

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1],
                                    v.cpu()] for k, v in state_dict.items()])
            # torch.save(weights, self.arg.model_saved_name + '-' + str(epoch) + '-' + str(int(self.global_step)) + '.pt')
            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch) + '.pt')


        # if epoch == 99:
        #     self.checkpoint()



    def eval(self, epoch, loader_name=['test'], wrong_file=None, result_file=None):
        # number of bins

        # if wrong_file is not None:
        #     f_w = open(wrong_file, 'w')
        # if result_file is not None:
        #     f_r = open(result_file, 'w')
        self.model.eval()
        # self.print_log('Eval epoch: {}'.format(epoch + 1))
        print("======================================================================================")
        for ln in loader_name:
            total_loss_value = []
            loss_value_yaw = []
            loss_value_pitch = []
            loss_value_roll = []
            score_frag_yaw = []
            score_frag_pitch = []
            score_frag_roll = []
            # right_num_total = 0
            # total_num = 0
            # loss_total = 0
            step = 0

            yaw_error = .0
            pitch_error = .0
            roll_error = .0

            total = 0
            total_yaw = []
            total_pitch = []
            total_roll = []
            total_mae = 0
            process = tqdm(self.data_loader[ln])
            total_embed = []
            total_embed_label = []
            total_feat = []
            # total_grid = []
            total_matrix = []
            mae_error = []
            maev_error = 0
            l_error = 0
            d_error = 0
            f_error = 0
            for batch_idx, (img, ref, matrix_label,  euler_label, ref_matrix_label, ref_euler_label, index) in enumerate(process):
                with torch.no_grad():
                    total += img.size(0)
                    img = Variable(img.float().cuda(self.output_device), requires_grad=False)
                    euler_label = Variable(euler_label.float().cuda(self.output_device), requires_grad=False)
                    matrix_label = Variable(matrix_label.float().cuda(self.output_device), requires_grad=False)
                    
                    k = 5
                    ref = [Variable(ref[i].float().cuda(self.output_device), requires_grad=False) for i in range(k)]
                    ref_euler_label = [Variable(ref_euler_label[i].float().cuda(self.output_device), requires_grad=False) for i in range(k)]
                    ref_matrix_label = [Variable(ref_matrix_label[i].float().cuda(self.output_device), requires_grad=False) for i in range(k)]
                
                    label_yaw = euler_label[:, 0]
                    label_pitch = euler_label[:, 1]
                    label_roll = euler_label[:, 2]
                    
                    
                    d_label = [compute_distance(matrix_label, ref_matrix_label[i]) for i in range(k)]
                    d_label = [Variable(d_label[i].float().cuda(self.output_device), requires_grad=False) for i in range(k)]

 

                    pre_matrix = self.model(img, extract=True)
                    # pre_matrix, pre_matrix_embed, x4= self.model(matrix_label, ref_euler_label, extract=True)
                    # _, s, _ = torch.svd(pre_matrix.view(-1,3,3))
                    # print(torch.sum(s,dim=-1))
                    pre_matrix = torch.clamp(pre_matrix,min=-1,max=1)
                    pre_matrix = tools.symmetric_orthogonalization(pre_matrix)
                    
                    pre_pitch, pre_yaw, pre_roll = utils.mat_to_euler(pre_matrix.view(-1,3,3).detach())
                    pre_yaw = torch.rad2deg(pre_yaw).cpu()
                    pre_pitch = torch.rad2deg(pre_pitch).cpu()
                    pre_roll = torch.rad2deg(pre_roll).cpu()

                    # Mean absolute error each angle
                    yaw_error += torch.sum(torch.abs(pre_yaw - label_yaw.cpu()))
                    pitch_error += torch.sum(torch.abs(pre_pitch - label_pitch.cpu()))
                    roll_error += torch.sum(torch.abs(pre_roll - label_roll.cpu()))
                    mae_error.append(torch.abs(pre_yaw - label_yaw.cpu()) +torch.abs(pre_pitch - label_pitch.cpu()) )
                    l_error += torch.sum(torch.rad2deg(torch.acos(torch.clamp(torch.sum(pre_matrix.view(-1,3,3)[:,:,0]*matrix_label.view(-1,3,3)[:,:,0],dim=-1),min=-0.9999,max=0.9999))))
                    d_error += torch.sum(torch.rad2deg(torch.acos(torch.clamp(torch.sum(pre_matrix.view(-1,3,3)[:,:,1]*matrix_label.view(-1,3,3)[:,:,1],dim=-1),min=-0.9999,max=0.9999))))
                    f_error += torch.sum(torch.rad2deg(torch.acos(torch.clamp(torch.sum(pre_matrix.view(-1,3,3)[:,:,2]*matrix_label.view(-1,3,3)[:,:,2],dim=-1),min=-0.9999,max=0.9999))))
                    maev_error += torch.sum(compute_maev(pre_matrix, matrix_label))
                    # Total MAE
                    # total_mae += (yaw_error + pitch_error + roll_error) / 3

                    # loss = self.loss(output, label)


                    # loss_value.append(loss.data.item())
                    loss_value_yaw.append(yaw_error.data.item())
                    loss_value_pitch.append(pitch_error.data.item())
                    loss_value_roll.append(roll_error.data.item())

                    # total_loss_value.append(total_mae.data.item())
                    # if batch_idx == 0:
                    #     total_img = img.detach().cpu().numpy()
                    #     total_embed = pre_matrix_embed.detach().cpu().numpy()
                    #     total_embed_label = matrix_embed_label.detach().cpu().numpy()
                    #     total_feat = x4.flatten(1).detach().cpu().numpy()
                    #     total_matrix = matrix_label.detach().cpu().numpy()
                    # else:
                    #     total_img = np.concatenate((total_img, img.detach().cpu().numpy()),axis=0)
                    #     # total_grid = np.concatenate((total_grid, grid.detach().cpu().numpy()),axis=0)
                    #     total_matrix = np.concatenate((total_matrix, matrix_label.detach().cpu().numpy()),axis=0)
                    #     total_embed = np.concatenate((total_embed, pre_matrix_embed.detach().cpu().numpy()),axis=0)
                    #     total_embed_label = np.concatenate((total_embed_label, matrix_embed_label.detach().cpu().numpy()),axis=0)
                    #     total_feat = np.concatenate((total_feat, x4.flatten(1).detach().cpu().numpy()),axis=0)

            total_loss = np.mean(total_loss_value)

            self.print_log('\tMAEV {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), (l_error + d_error + f_error) /(3*total)))
            self.print_log('\tLeft MAEV {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), (l_error) /(total)))
            self.print_log('\tDown MAEV {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), (d_error) /(total)))
            self.print_log('\tFront MAEV {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), (f_error) /(total)))
            self.print_log('\tMAE {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), (yaw_error + pitch_error + roll_error) /(3*total)))
            self.print_log('\tMAE yaw {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), yaw_error / total))
            self.print_log('\tMAE pitch {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]),pitch_error / total))
            self.print_log('\tMAE roll {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), roll_error / total))
    
        # return np.array(total_img), np.array(total_embed) , np.array(total_embed_label), np.array(total_feat), np.array(total_matrix)

    def start(self):
      
        # self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
        # for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
        
        #     save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
        #             epoch + 1 == self.arg.num_epoch)

        #     self.train(epoch, save_model=save_model)

        #     self.eval(
        #         epoch,
        #         loader_name=['test1'])
            
        #     self.eval(
        #         epoch,
        #         loader_name=['test2'])


        # print('best accuracy: ', self.best_acc, ' model_name: ', self.arg.model_saved_name)


        weights = torch.load(self.arg.weights_file_extract)
        print(self.arg.weights_file_extract)
        self.model.load_state_dict(weights)
        self.eval(epoch=0, loader_name=['test1'])
        self.eval(epoch=0, loader_name=['test2'])
        # np.savez('./grid_test1.npz', mat=total_matrix1, img=total_img1, grid=total_grid1)
        # np.savez('./embed_test2.npz', img=total_img2, embed=total_embed2, embed_label=total_embed_label2, feat=total_feat2, mat=total_mat2)
        # else:
        #     best_epoch = []
        #     for file in os.listdir(self.arg.weights_file):
            
        #         if file.endswith(".pt") and not 'checkpoint' in file:
        #             self.arg.weights = os.path.join(self.arg.weights_file, file)
                
        #             if self.arg.weights:
        #                 self.global_step = int(arg.weights[:-3].split('-')[-1])
        #                 self.print_log('Load weights from {}.'.format(self.arg.weights))
        #                 if '.pkl' in self.arg.weights:
        #                     with open(self.arg.weights, 'r') as f:
        #                         weights = pickle.load(f)
        #                 else:
        #                     weights = torch.load(self.arg.weights)

        #                 weights = OrderedDict(
        #                     [[k.split('module.')[-1],
        #                         v.cuda(self.output_device)] for k, v in weights.items()])

                    
        #                 try:
        #                     self.model.load_state_dict(weights)
        #                 except:
        #                     state = self.model.state_dict()
        #                     diff = list(set(state.keys()).difference(set(weights.keys())))
        #                     print('Can not find these weights:')
        #                     for d in diff:
        #                         print('  ' + d)
        #                     state.update(weights)
        #                     self.model.load_state_dict(state)
                            
                        
        #                 self.print_log('Model:   {}.'.format(self.arg.model))
        #                 self.print_log('Weights: {}.'.format(self.arg.weights))
        #                 self.eval(epoch=0, loader_name=['test1'], wrong_file=wf,
        #                         result_file=rf)
        #                 self.eval(epoch=0, loader_name=['test1'], wrong_file=wf,
        #                         result_file=rf)
            #             best_epoch.append((self.arg.weights, mae))
            # best_epoch = sorted(best_epoch, key=lambda x: x[1])
            # for i in range(10):
            #     print(f'Top {i}: ' + str(best_epoch[i]))

        self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def matrix_loss(matrix, matrix_label):
    matrix = matrix.view(-1,3,3)
    matrix_label = matrix_label.view(-1,3,3)
    v1 = matrix[:,:,0]
    v2 = matrix[:,:,1]
    v3 = matrix[:,:,2]
    
    v1_label = matrix_label[:,:,0]
    v2_label = matrix_label[:,:,1]
    v3_label = matrix_label[:,:,2]
    
    mse_loss = (torch.sum((v1 - v1_label)**2,dim=1) 
                + torch.sum((v2 - v2_label)**2,dim=1)
                + torch.sum((v3 - v3_label)**2,dim=1)).mean()

    v1 = v1 / (torch.norm(v1,dim=1, keepdim=True)+1e-10)
    v2 = v2 / (torch.norm(v2,dim=1, keepdim=True)+1e-10)
    v3 = v3 / (torch.norm(v3,dim=1, keepdim=True)+1e-10)
    
    ortho_loss = torch.mean(torch.abs(torch.sum(v1 * v2, dim=1))**2)\
                + torch.mean(torch.abs(torch.sum(v2 * v3, dim=1))**2)\
                + torch.mean(torch.abs(torch.sum(v1 * v3, dim=1))**2)
    return mse_loss + 0.1*ortho_loss


def compute_distance(matrix1, matrix2):
    matrix1 = matrix1.view(-1,3,3)
    matrix2 = matrix2.view(-1,3,3)
    d = torch.bmm(matrix1, matrix2)
    d = torch.acos(torch.clamp((d[:,0,0] + d[:,1,1] + d[:,2,2])/3,min=-0.9999, max=0.9999))
    return d

def compute_maev(matrix1,matrix2):
    matrix1 = matrix1.view(-1,3,3)
    matrix2 = matrix2.view(-1,3,3)
    maev = torch.sum(torch.abs(matrix1 - matrix2),dim=(1,2))/3
    return maev

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