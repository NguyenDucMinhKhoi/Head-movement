import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from graph.ntu_rgb_d import Graph, Graph1
from torch import sin, cos
from torch.nn.modules.utils import _pair
# from functions import deform_conv_function
import torchvision
import math


class ConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, activation='gelu'):
        super(ConvBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ConvPair2d(nn.Module):
    def __init__(self, in_channels, out_channels, size, padding=1, kernel_size=3, is_input=False):
        super(ConvPair2d, self).__init__()
        self.conv1 = ConvBlock2d(in_channels=in_channels,out_channels=in_channels,padding=padding,kernel_size=kernel_size)
        self.conv2 = ConvBlock2d(in_channels=in_channels,out_channels=out_channels,padding=padding,kernel_size=kernel_size,activation='sigmoid')
        if is_input:
            self.att1 = Attention(in_channels, (in_channels)//4)
        else:
            self.att1 = Attention(in_channels-3, (in_channels-3)//4)
        self.fc = nn.Linear(size*3,2)
        self.tanh = nn.Tanh()
        self.bn = nn.BatchNorm1d(2)
        self.att2 = SpatialAttention(3)
    def forward(self, x, a, grid,  return_att=True):
        _, att1 = self.att1(a)
        x = self.conv1(x)
        x = self.conv2(x)
        # x = torch.cat((g,x),dim=1)
        
        _, att2 = self.att2(torch.cat((grid,x),dim=1))
        
        # x = att1*att2*x

        t = torch.cat((grid, x), dim=1).flatten(1)
        t = self.tanh(self.bn(self.fc(t))).unsqueeze(-1).unsqueeze(-1)*0.5
        if return_att:
            return x, t, att1*att2
        return x, t

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    

    def forward(self, x):
        bs, c, h, w = x.size()
        x = nn.AvgPool2d(kernel_size=(h,w),stride=(h,w))(x)
        return x 

class GlobalMaxPool(nn.Module):
    def __init__(self):
        super(GlobalMaxPool, self).__init__()

    

    def forward(self, x):
        bs, c, h, w = x.size()
        x = nn.MaxPool2d(kernel_size=(h,w),stride=(h,w))(x)
        return x 

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, embed_channels):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, embed_channels),
            nn.Sigmoid(),
            nn.Linear(embed_channels, in_channels)
            )
        self.pool = GlobalAvgPool()

    def forward(self, x):
        # print(x.shape)
        att = self.pool(x)

        # print(att.shape)
        att = self.mlp(torch.flatten(att,start_dim=1))
        scale = nn.Sigmoid()(att).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.in_channels = in_channels
        self.conv1 = ConvBlock2d(in_channels=in_channels, out_channels=1, kernel_size=7,padding=3, activation='no')
        # self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,padding=1)
        self.conv2 =  ConvBlock2d(in_channels=in_channels, out_channels=1, kernel_size=7,padding=3, activation='no')
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(1)
        # self.conv1 = nn.Sequential(ConvBlock2d(in_channels=in_channels, out_channels=in_channels//4, kernel_size=3,padding=1, activation='sigmoid'), 
        #                            ConvBlock2d(in_channels=in_channels//4, out_channels=in_channels, kernel_size=3,padding=1, activation='sigmoid'))
        # self.conv2 = nn.Sequential(ConvBlock2d(in_channels=in_channels, out_channels=in_channels//4, kernel_size=3,padding=1, activation='sigmoid'), 
        #                            ConvBlock2d(in_channels=in_channels//4, out_channels=in_channels, kernel_size=3,padding=1, activation='no'))
      
        # self.conv3 = ConvBlock2d(in_channels=in_channels, out_channels=1, kernel_size=1,padding=0, activation='no')
    def forward(self, x):

        att = self.conv1(x)*self.conv2(x)
        # att = self.conv3(att)
        att = self.bn(att)
        att = self.sigmoid(att)
        # att = self.conv3(att)
        # att = self.sigmoid(att)
        return x * att, att
    

class Attention_pure(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5
        self.attn_drop = nn.Dropout()
        self.proj_drop = nn.Dropout()
        self.conv1 = nn.Conv2d(dim, 3 * dim, 1)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.conv1(x)
        x = x.flatten(start_dim=2).transpose(1,2)
        B, N, C = x.shape
        C = int(C // 3)
        qkv = x.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        return x
class Attention(nn.Module):
    def __init__(self, in_channels, embed_channels):
        super(Attention, self).__init__()
        self.s_attention = SpatialAttention(in_channels)
        self.c_attention = ChannelAttention(in_channels, embed_channels)

    def forward(self,x):
        x = self.c_attention(x)
        x, att = self.s_attention(x)
        return x

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.conv11 = ConvBlock2d(in_channels=3,out_channels=6)
        self.conv12 = ConvBlock2d(in_channels=6,out_channels=6)
        self.conv13 = ConvBlock2d(in_channels=3,out_channels=6, kernel_size=1, padding=0, activation='no')

        self.conv21 = ConvBlock2d(in_channels=6,out_channels=18)
        self.conv22 = ConvBlock2d(in_channels=18,out_channels=36)
        self.conv23 = ConvBlock2d(in_channels=6,out_channels=36, kernel_size=1, padding=0, activation='no')

        self.conv31 = ConvBlock2d(in_channels=36,out_channels=72)
        self.conv32 = ConvBlock2d(in_channels=72,out_channels=144)
        self.conv33 = ConvBlock2d(in_channels=36,out_channels=144, kernel_size=1, padding=0, activation='no')

        # self.conv41 = ConvBlock2d(in_channels=36,out_channels=48)
        # self.conv42 = ConvBlock2d(in_channels=48,out_channels=144)
        # self.conv43 = ConvBlock2d(in_channels=36,out_channels=144, kernel_size=1, padding=0, activation='no')
        self.dropout= nn.Dropout()
    def forward(self, x):
        x11 = self.conv11(x)
        x11 = self.conv12(x11)
        x12 = self.conv13(x)
        x1 = x11 + x12
        x1 = self.dropout(x1)
        x1_pool = self.pool(x1)
        
        x21 = self.conv21(x1_pool)
        x21 = self.conv22(x21)
        x22 = self.conv23(x1_pool)
        x2 = x21 + x22
        x2 = self.dropout(x2)
        x2_pool = self.pool(x2)

        x31 = self.conv31(x2_pool)
        x31 = self.conv32(x31)
        x32 = self.conv33(x2_pool)
        x3 = x31 + x32
        x3 = self.dropout(x3)
        
        # x3_pool = self.pool(x3)

        # x41 = self.conv41(x3_pool)
        # x41 = self.conv42(x41)
        # x42 = self.conv43(x3_pool)
        # x4 = x41 + x42
    
        return x1, x2, x3




    
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.base_model = BaseModel()
        
        self.fc = nn.Linear(144,9)
        # self.conv = nn.Conv1d(in_channels=144//3,out_channels=3,kernel_size=1,padding=0)
        # self.bn = nn.BatchNorm1d(num_features=9)
        # self.tanh = nn.Tanh()
        self.att = Attention(144,144//4)
        self.dropout = nn.Dropout()
        self.pool = GlobalMaxPool()  
        self.tanh = nn.Tanh()
        self.fc_u = nn.Sequential(nn.Linear(144,256),
                                  nn.BatchNorm1d(256),
                                  nn.GELU(),
                                  nn.Dropout(),
                                  nn.Linear(256,256),
                                  nn.BatchNorm1d(256),
                                  nn.GELU(),
                                  nn.Dropout(),
                                  nn.Linear(256,3),
                                  nn.BatchNorm1d(3),
                                  nn.Tanh()
                                )
    
   
    def forward(self, x, ref=None, ref_pose=None, x_pose=None,k=5, extract=False,mode='eval'):
        
        if mode == 'train':
            bs,_,_,_ = x[0].shape
            me = []
            m = []
            p = []
            for i in range(9):
                x1, x2, x4 = self.base_model(x[i])
                x4 = self.att(x4)
                x5 = self.pool(x4).flatten(1)
                x5 = x5.view(-1,3,144//3)
                u = 10*(self.fc_u(x5.flatten(1))) + 10.1
                x5_unnorm = x4
                x5 = x5/(torch.norm(x5,dim=-1,keepdim=True)+1e-6)
                x5 = x5.flatten(1)
                me.append(x5)

                mi = self.fc(x5)
                m.append(mi)
                U, _, V = torch.svd(mi.view(-1,3,3))
            
                pi = torch.bmm(U,torch.diag_embed(u))
                pi = torch.bmm(pi,V.permute(0,2,1))
                p.append(pi)
       
          
            rm = []
            rme = []
            rp = []
            for _ in range(k):
                rm.append([])
                rme.append([])
                rp.append([])
            for i in range(k):
                for j in range(9):
                    _, _, r4 = self.base_model(ref[i][j])
                    r4 = self.att(r4)
                    r5 = self.pool(r4).flatten(1)
                    # r5 = self.dropout(r5)
                    r5 = r5.view(-1,3,144//3)
                    ru = 10*(self.fc_u(r5.flatten(1))) + 10.1
                    r5 = r5/(torch.norm(r5,dim=-1,keepdim=True)+1e-6)
                    r5 = r5.flatten(1)
                    rme[i].append(r5)
                    # rm.append(self.conv(r5.view(-1,3,144//3).permute(0,2,1)).flatten(1)) 
                    r_m = self.fc(r5)
                    # r_m = r_m/torch.norm(r_m, dim=-1,keepdim=True)
                    # r_m 
                    U, _, V = torch.svd(r_m.view(-1,3,3))
                    r_p = torch.bmm(U,torch.diag_embed(ru))
                    r_p = torch.bmm(r_p, V.permute(0,2,1))
                    rm[i].append(r_m)
                    rp[i].append(r_p)
              


            projector = []
            for i in range(len(ref)):
                P = [r.unsqueeze(1) for r in rme[i]]
                P = torch.cat(P,dim=1)
                M = ref_pose[i].unsqueeze(1)
                M = M.repeat(1,9,1)
                # A = torch.linalg.pinv(P) @ M
                A = torch.linalg.lstsq(P,M)[0]
                projector.append(A)

            P = [in_im.unsqueeze(1) for in_im in me]
            P = torch.cat(P,dim=1)
            M = x_pose.unsqueeze(1)
            M = M.repeat(1,9,1)
            in_projector = torch.linalg.lstsq(P,M)[0]
            
            x_dis_loss = 0
            id_mat = torch.eye(3).to(x[0].device)
            id_mat = id_mat.view(-1,3,3)
            id_mat = id_mat.repeat(bs, 1,1) 
            ref_mat_dis = [[] for _ in range(len(ref))]
            d = [[] for _ in range(len(ref))]
            reverse_d = [[] for _ in range(len(ref))]
            for i in range(len(ref)):
                diff_m = torch.bmm(ref_pose[i].view(-1,3,3), x_pose.view(-1,3,3).permute(0,2,1))
               
                for j in range(9):
                    d[i].append(compute_distance_mat(torch.bmm(me[j].unsqueeze(1),projector[i]).flatten(1), torch.bmm(rme[i][j].unsqueeze(1),projector[i]).flatten(1)))
                     
                    em = torch.bmm(diff_m,me[j].view(-1,3,144//3))
                    em = em.flatten(1).unsqueeze(1)
                #  
                    # print(torch.mean((self.fc(em.flatten(1).detach())-ref_pose[i])**2/9), torch.mean((torch.bmm(em.detach(),projector[i]).flatten(1) - ref_pose[i])**2/9))

                    m_dis = torch.bmm(em,projector[i]).flatten(1)
                    # print(torch.mean((self.fc(em.detach().flatten(1))-ref_pose[i])**2/9), torch.mean((m_dis - ref_pose[i])**2/9))

                    # m_dis = m_dis.view(-1,3,3)
                    # # m_dis = m_dis / torch.norm(m_dis, dim=-1,keepdim=True)
                    # m_dis = torch.bmm(m_dis.permute(0,2,1), ref_pose[i].view(-1,3,3))
                    # m_dis = (torch.norm(m_dis - id_mat) )**2
                    # x_dis_loss += torch.mean(m_dis)
                    ref_mat_dis[i].append(m_dis)
            # x_dis_loss /= len(ref)*9

            ref_dis_loss = 0
            in_mat_dis = []
            in_mat_dis = [[] for _ in range(len(ref))] 
            for i in range(len(ref)):
                diff_m = torch.bmm(x_pose.view(-1,3,3), ref_pose[i].view(-1,3,3).permute(0,2,1))
               
                for j in range(9):
               
                    rem = torch.bmm(diff_m,rme[i][j].view(-1,3,144//3))

                    rem = rem.flatten(1).unsqueeze(1)
                    r_dis = torch.bmm(rem, in_projector).flatten(1)
                    # r_dis = r_dis.view(-1,3,3)
                    in_mat_dis[i].append(r_dis)
                    reverse_d[i].append(compute_distance_mat(torch.bmm(me[j].unsqueeze(1),in_projector).flatten(1), torch.bmm(rme[i][j].unsqueeze(1),in_projector).flatten(1)))
                    # r_dis = r_dis/torch.norm(r_dis,dim=-1,keepdim=True)
                    # r_dis = torch.bmm(r_dis.permute(0,2,1), x_pose.view(-1,3,3))
                    # r_dis = (torch.norm(r_dis - id_mat))**2
                    # ref_dis_loss += torch.mean(r_dis)
            # ref_dis_loss /= len(ref)*9
            # dis_loss = 1/2*(x_dis_loss + ref_dis_loss)
            return m, p,  me, rm,  rp, rme, ref_mat_dis, in_mat_dis, d, reverse_d
        else:
            bs,_,_,_ = x.shape
            x1, x2, x4 = self.base_model(x)
            x4 = self.att(x4)
            x5 = self.pool(x4).flatten(1)
            # x5 = self.dropout(x5)
            x5 = x5.view(-1,3,144//3)
            u = 10*(self.fc_u(x5.flatten(1))) + 10.1
            # u = torch.exp(torch.clamp(self.fc_u(x5.flatten(1)), max=10))
            x5_unnorm = x4
            x5 = x5/(torch.norm(x5,dim=-1,keepdim=True)+1e-6)
            x5 = x5.flatten(1)
            me = x5

            m = self.fc(x5)
            U, _, V = torch.svd(m.view(-1,3,3))
          
            p = torch.bmm(U,torch.diag_embed(u))
            p = torch.bmm(p,V.permute(0,2,1))
            x5 = x5.view(-1,3,144//3)
            
                
            if extract:
                return m, me, x4
            else:
                return m

def compute_distance(feat1, feat2):
    bs, _, _ = feat1.shape 
    feat2 =feat2.permute(0,2,1)  
    # id_matrix = torch.eye(3).unsqueeze(0).repeat(bs,1,1).to(feat1.device)
    d = torch.bmm(feat1, feat2)
    # d = torch.sqrt((d[:,0,0]-1)**2 + (d[:,1,1]-1)**2 + (d[:,2,2]-1)**2)
    d = torch.sqrt(2*(3-(d[:,0,0]+d[:,1,1]+d[:,2,2]))) # Equation 3-6 in the paper

def compute_distance_mat(matrix1, matrix2):
    bs, _ = matrix1.shape
    matrix1 = matrix1.view(-1,3,3)
    matrix2 = matrix2.view(-1,3,3)
    # id_matrix = torch.eye(3).unsqueeze(0).repeat(bs,1,1).to(matrix1.device)

    # d = torch.norm(id_matrix - torch.bmm(matrix1, matrix2.permute(0,2,1)),dim=(1,2))
    d = torch.bmm(matrix1, matrix2.permute(0,2,1))
    # d = torch.sqrt((d[:,0,0] - 1)**2 + (d[:,1,1]-1)**2 + (d[:,2,2]-1)**2)
    d = torch.sqrt(torch.clamp(2*(3-(d[:,0,0]+d[:,1,1]+d[:,2,2])),min=1e-6))
    return d