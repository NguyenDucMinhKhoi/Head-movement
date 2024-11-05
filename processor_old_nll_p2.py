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
from model.ref_dis_noise_l2_nll_v2 import Model
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
            train_dataset = dataset.BIWI(self.arg.train_data_path, self.arg.train_file_name, self.arg.train_data_path, self.arg.train_file_name, mode='train')
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

        # if self.arg.test_dataset2 == 'Pose_300W_LP':
        #     test_dataset2 = dataset.Pose_300W_LP(self.arg.test_data_path2, self.arg.test_file_name2, self.arg.train_data_path, self.arg.train_file_name)
        # elif self.arg.test_dataset2 == 'AFLW2000':
        #     test_dataset2 = dataset.AFLW2000(self.arg.test_data_path2, self.arg.test_file_name2, self.arg.train_data_path, self.arg.train_file_name)
        # elif self.arg.test_dataset2 == 'BIWI':
        #     test_dataset2 = dataset.BIWI(self.arg.test_data_path2, self.arg.test_file_name2, self.arg.train_data_path, self.arg.train_file_name)
        # else:
        #     print('Error: not a valid dataset name')
        #     sys.exit()

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

        # self.data_loader['test2'] = torch.utils.data.DataLoader(
        #     dataset=test_dataset2,
        #     batch_size=self.arg.test_batch_size,
        #     shuffle=False,
        #     num_workers=self.arg.num_worker,
        #     drop_last=False,
        #     worker_init_fn=init_seed)

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
        #     # if '.pkl' in self.arg.weights:
        #     #     with open(self.arg.weights, 'r') as f:
        #     #         weights = pickle.load(f)
        #     # else:
        #     #     weights = torch.load(self.arg.weights)

        #     # weights = OrderedDict(
        #     #     [[k.split('module.')[-1],
        #     #       v.cuda(output_device)] for k, v in weights.items()])

        #     # keys = list(weights.keys())
        #     # for w in self.arg.ignore_weights:
        #     #     for key in keys:
        #     #         if w in key:
        #     #             if weights.pop(key, None) is not None:
        #     #                 self.print_log('Sucessfully Remove Weights: {}.'.format(key))
        #     #             else:
        #     #                 self.print_log('Can Not Remove Weights: {}.'.format(key))

        #     # try:
        #     #     self.model.load_state_dict(weights)
        #     # except:
        #     #     state = self.model.state_dict()
        #     #     diff = list(set(state.keys()).difference(set(weights.keys())))
        #     #     print('Can not find these weights:')
        #     #     for d in diff:
        #     #         print('  ' + d)
        #     #     state.update(weights)
        #     #     self.model.load_state_dict(state)

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
        for batch_idx, ( img, ref, matrix_label, euler_label, ref_matrix_label,  ref_euler_label, index
) in enumerate(process):
            # get data
            # torch.autograd.set_detect_anomaly(True)
            total += img.size(0)
            img = Variable(img.float().cuda(self.output_device), requires_grad=False)
            euler_label = Variable(euler_label.float().cuda(self.output_device), requires_grad=False)
            matrix_label = Variable(matrix_label.float().cuda(self.output_device), requires_grad=False)
            
            k = 10
            ref = [Variable(ref[i].float().cuda(self.output_device), requires_grad=False) for i in range(k)]
            ref_euler_label = [Variable(ref_euler_label[i].float().cuda(self.output_device), requires_grad=False) for i in range(k)]
            ref_matrix_label = [Variable(ref_matrix_label[i].float().cuda(self.output_device), requires_grad=False) for i in range(k)]
            timer['dataloader'] += self.split_time()
         
            label_yaw = euler_label[:, 0]
            label_pitch = euler_label[:, 1]
            label_roll = euler_label[:, 2]
            
            
            d_label = [compute_distance(matrix_label, ref_matrix_label[i]) for i in range(k)]
            d_label = [Variable(d_label[i].float().cuda(self.output_device), requires_grad=False) for i in range(k)]
            # ref_d_label = [compute_distance(ref_matrix_label[i], ref_matrix_label[j]) for i in range(k-1) for j in range(i+1,k)]
            # ref_d_label = [Variable(ref_d_label[i].float().cuda(self.output_device), requires_grad=False) for i in range(len(ref_d_label))]
            # matrix_embed_label = Variable(matrix_embed_label.float().cuda(self.output_device), requires_grad=False)
            # ref_matrix_embed_label = [Variable(ref_matrix_embed_label[i].float().cuda(self.output_device), requires_grad=False) for i in range(k)]

            # forward
            pre_matrix, p, pre_matrix_embed, ref_matrix, rp, ref_matrix_embed, d = self.model(img, ref, ref_matrix_label, x_pose=matrix_label, k=k)
            
            d_loss = 0
            o_loss = ortho_loss(pre_matrix_embed.view(-1,3,144//3))
            nll = nll_loss(p, matrix_label)
            for i in range(k):
                d_loss += self.reg_criterion(d[i], d_label[i])
                o_loss += ortho_loss(ref_matrix_embed[i].view(-1,3,144//3))
                nll += nll_loss(rp[i], ref_matrix_label[i])

            # for i in range(len(ref_d_label)):
            #         d_loss += self.reg_criterion(ref_d[i], ref_d_label[i])

            # embed_loss = MMD_loss(pre_matrix_embed,matrix_embed_label)
            # for i in range(k):
            #     embed_loss += MMD_loss(ref_matrix_embed[i], ref_matrix_embed_label[i])
            
            
            beta = 0.5
            pose_loss = matrix_loss(pre_matrix, matrix_label) 
            for i in range(k): 
                pose_loss += matrix_loss(ref_matrix[i], ref_matrix_label[i]) 
            
            # Total loss
            
            loss = pose_loss/(k+1)+ beta*d_loss/k + 1e-6*o_loss/(k+1) + 0.5*nll/(k+1)
            
    
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            # gradient clip
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()
            pre_matrix = torch.clamp(pre_matrix.detach(),min=-1,max=1)
            pre_matrix = pre_matrix.view(-1,3,3)
            pre_matrix = pre_matrix/(torch.norm(pre_matrix,dim=-1,keepdim=True) + 1e-6)
            pre_matrix = pre_matrix.flatten(1)
            # print(torch.mean(pre_matrix.view(-1,3,3),dim=0))
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
           
            for batch_idx, ( img, ref, matrix_label, euler_label, ref_matrix_label,  ref_euler_label, index) in enumerate(process):
                with torch.no_grad():
                    total += img.size(0)
                    img = Variable(
                        img.float().cuda(self.output_device),
                        requires_grad=False,
                        volatile=True)
                    euler_label = Variable(
                        euler_label.float().cuda(self.output_device),
                        requires_grad=False,
                        volatile=True)
                    
                    matrix_label = Variable(
                        matrix_label.float().cuda(self.output_device),
                        requires_grad=False,
                        volatile=True)

                       
                    # ref = Variable(ref.float().cuda(self.output_device), requires_grad=False)
                    # ref_euler_label = Variable(ref_euler_label.float().cuda(self.output_device), requires_grad=False)
                    # ref_matrix_label = Variable(ref_matrix_label.float().cuda(self.output_device), requires_grad=False)
                    label_yaw = euler_label[:, 0]
                    label_pitch = euler_label[:, 1]
                    label_roll = euler_label[:, 2]


                    pre_matrix=  self.model(img)
                    pre_matrix = torch.clamp(pre_matrix,min=-1,max=1)
                    pre_matrix = pre_matrix.view(-1,3,3)
                    pre_matrix = pre_matrix/(torch.norm(pre_matrix,dim=-1,keepdim=True))
                    pre_matrix = pre_matrix.flatten(1)
                    pre_matrix = tools.symmetric_orthogonalization(pre_matrix)
                    pre_pitch, pre_yaw, pre_roll = utils.mat_to_euler(pre_matrix.view(-1,3,3).detach())
                    pre_yaw = torch.rad2deg(pre_yaw).cpu()
                    pre_pitch = torch.rad2deg(pre_pitch).cpu()
                    pre_roll = torch.rad2deg(pre_roll).cpu()

                    # Mean absolute error each angle
                    yaw_error += torch.sum(torch.abs(pre_yaw - label_yaw.cpu()))
                    pitch_error += torch.sum(torch.abs(pre_pitch - label_pitch.cpu()))
                    roll_error += torch.sum(torch.abs(pre_roll - label_roll.cpu()))

                    # Total MAE
                    # total_mae += (yaw_error + pitch_error + roll_error) / 3

                    # loss = self.loss(output, label)


                    # loss_value.append(loss.data.item())
                    # loss_value_yaw.append(yaw_error.data.item())
                    # loss_value_pitch.append(pitch_error.data.item())
                    # loss_value_roll.append(roll_error.data.item())

                    # total_loss_value.append(total_mae.data.item())



            # total_loss = np.mean(total_loss_value)


            self.print_log('\tMAE {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), (yaw_error + pitch_error + roll_error) /(3*total)))
            self.print_log('\tMAE yaw {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), yaw_error / total))
            self.print_log('\tMAE pitch {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]),pitch_error / total))
            self.print_log('\tMAE roll {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), roll_error / total))

        return total_mae

    def start(self):
      
        # self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
        
            save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                    epoch + 1 == self.arg.num_epoch)
            
            self.train(epoch, save_model=save_model)

            self.eval(
                epoch,
                loader_name=['test1'])
            
            # self.eval(
            #     epoch,
            #     loader_name=['test2'])


        print('best accuracy: ', self.best_acc, ' model_name: ', self.arg.model_saved_name)

        # elif self.arg.phase == 'test':
  
      
        #     if not self.arg.test_loop:
        #         weights = torch.load(self.arg.weights_file)
        #         self.model.load_state_dict(weights)
        #         self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
        #     else:
        #         best_epoch = []
        #         for file in os.listdir(self.arg.weights_file):
                
        #             if file.endswith(".pt") and not 'checkpoint' in file:
        #                 self.arg.weights = os.path.join(self.arg.weights_file, file)
                  
        #                 if self.arg.weights:
        #                     self.global_step = int(arg.weights[:-3].split('-')[-1])
        #                     self.print_log('Load weights from {}.'.format(self.arg.weights))
        #                     if '.pkl' in self.arg.weights:
        #                         with open(self.arg.weights, 'r') as f:
        #                             weights = pickle.load(f)
        #                     else:
        #                         weights = torch.load(self.arg.weights)

        #                     weights = OrderedDict(
        #                         [[k.split('module.')[-1],
        #                           v.cuda(self.output_device)] for k, v in weights.items()])

                       
        #                     try:
        #                         self.model.load_state_dict(weights)
        #                     except:
        #                         state = self.model.state_dict()
        #                         diff = list(set(state.keys()).difference(set(weights.keys())))
        #                         print('Can not find these weights:')
        #                         for d in diff:
        #                             print('  ' + d)
        #                         state.update(weights)
        #                         self.model.load_state_dict(state)
                                
                           
        #                     self.print_log('Model:   {}.'.format(self.arg.model))
        #                     self.print_log('Weights: {}.'.format(self.arg.weights))
        #                     mae = self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf,
        #                             result_file=rf)
        #                     best_epoch.append((self.arg.weights, mae))
        #         best_epoch = sorted(best_epoch, key=lambda x: x[1])
        #         for i in range(10):
        #             print(f'Top {i}: ' + str(best_epoch[i]))

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

def compute_distance(matrix1, matrix2):
    bs, _ = matrix1.shape
    matrix1 = matrix1.view(-1,3,3)
    matrix2 = matrix2.view(-1,3,3)
    # id_matrix = torch.eye(3).unsqueeze(0).repeat(bs,1,1).to(matrix1.device)

    # d = torch.norm(id_matrix - torch.bmm(matrix1, matrix2.permute(0,2,1)),dim=(1,2))
    d = torch.bmm(matrix1, matrix2.permute(0,2,1))
    # d = torch.sqrt((d[:,0,0] - 1)**2 + (d[:,1,1]-1)**2 + (d[:,2,2]-1)**2)
    d = torch.sqrt(torch.clamp(2*(3-(d[:,0,0]+d[:,1,1]+d[:,2,2])),min=1e-6))
    return d
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

    # v1 = v1 / (torch.norm(v1,dim=1, keepdim=True)+1e-10)
    # v2 = v2 / (torch.norm(v2,dim=1, keepdim=True)+1e-10)
    # v3 = v3 / (torch.norm(v3,dim=1, keepdim=True)+1e-10)
    
    ortho_loss = torch.mean(torch.abs(torch.sum(v1 * v2, dim=1))**2)\
                + torch.mean(torch.abs(torch.sum(v2 * v3, dim=1))**2)\
                + torch.mean(torch.abs(torch.sum(v1 * v3, dim=1))**2)\
                + torch.mean(torch.abs(torch.sum(v1 * v1, dim=1)-1)**2)\
                + torch.mean(torch.abs(torch.sum(v2 * v2, dim=1)-1)**2)\
                + torch.mean(torch.abs(torch.sum(v3 * v3, dim=1)-1)**2)
    return mse_loss + 0.1*ortho_loss


def MMD_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source.flatten(1), target.flatten(1)], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # total0 = total.unsqueeze(0)
    # total1 = total.unsqueeze(1)
    # total0 = total
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

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


# def ortho_loss(vec):
    
#     id_matrix = torch.eye(3).to(vec.device)
#     return torch.mean(torch.abs(torch.matmul(vec,vec.permute(1,0)) - id_matrix)**2)

def ortho_loss(vec):
    bs,_,_ = vec.shape
    id_matrix = torch.eye(3).unsqueeze(0)
    id_matrix = id_matrix.repeat(bs,1,1).to(vec.device)
    return torch.mean(torch.sum(torch.abs(torch.bmm(vec,vec.permute(0,2,1)) - id_matrix)**2, dim=(-1,-2)))
    
    
def get_constant(F):
    bs, _, _ = F.shape
    _, S, _ = torch.linalg.svd(F, full_matrices=False)
    du = 0.001
    # S = torch.clamp(S,min=1e-6,max=100)
    u=torch.arange(-1,1,du).unsqueeze(0).repeat(bs,1).to(F.device)

    du = torch.FloatTensor([du]).unsqueeze(0).repeat(bs,1).to(F.device)

    f = 1/2*torch.special.i0(1/2*(S[:,0].unsqueeze(-1)-S[:,1].unsqueeze(-1))*(1-u))*torch.special.i0(1/2*(S[:,0].unsqueeze(-1)+S[:,1].unsqueeze(-1))*(1+u))*torch.exp(S[:,2].unsqueeze(-1)*u)
    return torch.sum(f*du,dim=-1)

def nll_loss(F,R):
    F = F.view(-1,3,3)
    # U, S, V = torch.linalg.svd(F, full_matrices=False)
    # Sm = torch.diag_embed(S)
    R = R.view(-1,3,3)
    a = get_constant(F)
    tr = torch.einsum('bii',torch.bmm(F.permute(0,2,1),R))
    # print(tr)
    return torch.mean(torch.log(torch.clamp(a,min=1e-6))-tr)



def mse(predict, target, w=None):
    if w is not None:
        return torch.mean(torch.norm(w*(predict - target),dim=-1)**2)
    else:
        return torch.mean(torch.norm(predict - target,dim=-1)**2)