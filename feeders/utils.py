import numpy as np
import torch
# from torch.utils.serialization import load_lua
# import torchfile

import os
import scipy.io as sio
import cv2
import math
from math import cos, sin

def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result

def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, tdx, tdy]
    pose_params = pre_pose_params[:5]
    return pose_params

def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params

def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d

def get_pt2d_origin_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d_origin']
    return pt2d

def get_roi_from_mat(mat_path):
    # Get ROI
    mat = sio.loadmat(mat_path)
    roi = mat['roi']
    return roi

def mse_loss(input, target):
    return torch.sum(torch.abs(input.data - target.data) ** 2)


def mat_to_euler(mat):
    # For intrinsic XYZ
    eps = 1e-7
    n1 = torch.FloatTensor([[1,0,0]]*len(mat))
    n2 = torch.FloatTensor([[0,1,0]]*len(mat))
    n3 = torch.FloatTensor([[0,0,1]]*len(mat))
    s1 = torch.sum(torch.cross(n1,n2,dim=1)*n3,dim=1)
    c1 = torch.sum(n1*n3, dim=1)
    offset = torch.atan2(s1,c1)
    
    C = torch.cat((n2,torch.cross(n1,n2,dim=1),n1), dim=1)
    C = C.view(-1,3,3)
    rot = torch.zeros_like(C)
    rot[:,0,0]=1
    rot[:,1,1] += c1
    rot[:,2,2] += c1
    rot[:,1,2] += s1
    rot[:,2,1] -= s1
    one  = torch.Tensor([1])
    if torch.cuda.is_available():
        C = C.cuda()
        rot = rot.cuda()
        one = one.cuda()
    angles = torch.zeros((len(mat),3))
    res = torch.matmul(C, mat)
    matrix_trans = torch.matmul(res, torch.matmul(torch.transpose(C, 1, 2), rot))
    

    matrix_trans[:, 2, 2] = torch.minimum(matrix_trans[:, 2, 2], one)
    matrix_trans[:, 2, 2] = torch.maximum(matrix_trans[:, 2, 2], -one)
    angles[:,1] = torch.acos(matrix_trans[:,2,2])

    safe1 = torch.abs(angles[:,1]) >= eps
    safe2 = torch.abs(angles[:,1] - torch.Tensor([math.pi])) >= eps
    safe = torch.logical_and(safe1,safe2) 
    pi = torch.Tensor([math.pi])

    if torch.cuda.is_available():    
        safe1 = safe1.cuda()
        safe2 = safe2.cuda()
        safe = safe.cuda()
        angles = angles.cuda()
        offset = offset.cuda()
        pi = pi.cuda()

    angles[:,1] +=offset
    angles[:,0] += safe.to(torch.float)*torch.atan2(matrix_trans[:,0, 2], -matrix_trans[:, 1, 2])
    angles[:,2] += safe.to(torch.float)*torch.atan2(matrix_trans[:,2, 0], matrix_trans[:,2, 1])
    angles[:,2] *= safe.to(torch.float)
    angles[:, 0] *= safe1.to(torch.float)
    angles[:, 0] += torch.logical_not(safe1).to(torch.float)*(torch.atan2(matrix_trans[:, 1, 0] - matrix_trans[:, 0, 1], matrix_trans[:, 0, 0] + matrix_trans[:, 1, 1]))
    angles[:, 0] *= safe2.to(torch.float)
    angles[:, 0] += torch.logical_not(safe1).to(torch.float)*(torch.atan2(matrix_trans[:, 1, 0] + matrix_trans[:, 0, 1], matrix_trans[:, 0, 0] - matrix_trans[:, 1, 1]))
    adjust = torch.logical_or(angles[:, 1] < -pi/2,  angles[:, 1] > pi/2)
    if torch.cuda.is_available():
        adjust = adjust.cuda()
    angles[:,0] += adjust.to(torch.float)*safe.to(torch.float)*pi
    angles[:,1] = torch.logical_not(torch.logical_and(adjust, safe)).to(torch.float)*angles[:,1] + torch.logical_and(adjust, safe).to(torch.float)*(2*offset - angles[:,1])
    angles[:,2] -= torch.logical_and(adjust, safe).to(torch.float)*pi
    angles +=  (angles < -pi).to(torch.float)*pi*2
    angles -=  (angles > pi).to(torch.float)*pi*2
    
    return angles[:,0], angles[:,1], angles[:,2]


def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * size
        face_y = tdy - 0.50 * size
    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,0,255),3)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x),int(y2+y1-face_y)),(0,0,255),3)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x),int(y1+y2-face_y)),(0,0,255),3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),2)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x),int(y1+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x),int(y2+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2+x1-face_x),int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(255,0,0),2)
    # Draw top in green
    cv2.line(img, (int(x3+x1-face_x),int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x2+x3-face_x),int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x),int(y3+y1-face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x),int(y3+y2-face_y)),(0,255,0),2)

    return img

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x3 = size * (cos(yaw) * cos(roll)) + tdx
    y3 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x1 = size * (sin(yaw)) + tdx
    y1 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img
