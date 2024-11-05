import scipy.io as sio
import pandas as pd
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import sys
import cv2
from moviepy.editor import *
import numpy as np
import argparse
from mtcnn.mtcnn import MTCNN


def get_args():
	parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
	                                             "and creates database for training.",
	                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--db", type=str, default='../raw_data/BIWI',
	                    help="path to database")
	parser.add_argument("--output", type=str, default='./BIWI',
	                    help="path to output database mat file")
	parser.add_argument("--img_size", type=int, default=64,
	                    help="output image size")
	parser.add_argument("--ad", type=float, default=0.4,
	                    help="enlarge margin")
	

	args = parser.parse_args()
	return args


def main():
	np.random.seed(0)
	args = get_args()
	mypath = args.db
	output_path = args.output
	img_size = args.img_size
	ad = args.ad

	isPlot = True
	detector = MTCNN()

	randFlag = np.zeros(24)
	randFlag[0:16] = 1
	randFlag = np.random.permutation(randFlag)

	print(randFlag)
	output_train_path = output_path+'_train.npz'
	output_test_path = output_path+'_test.npz'

	onlyfiles_png = []
	onlyfiles_txt = []

	for num in range(24):
		if num<9:
			mypath_obj = mypath+'/0'+str(num+1)
		else:
			mypath_obj = mypath+'/'+str(num+1)
		print(mypath_obj)
		onlyfiles_txt_temp = [f for f in listdir(mypath_obj) if isfile(join(mypath_obj, f)) and join(mypath_obj, f).endswith('.txt')]
		onlyfiles_png_temp = [f for f in listdir(mypath_obj) if isfile(join(mypath_obj, f)) and join(mypath_obj, f).endswith('.png')]
	
		onlyfiles_txt_temp.sort()
		onlyfiles_png_temp.sort()

		onlyfiles_txt.append(onlyfiles_txt_temp)
		onlyfiles_png.append(onlyfiles_png_temp)
	print(len(onlyfiles_txt))
	print(len(onlyfiles_png))
	
	out_imgs_train = []
	out_poses_train = []

	out_imgs_test = []
	out_poses_test = []
	
	# for i in range(len(onlyfiles_png)):
	for i in range(12):
		print('object %d' %i)
		
		mypath_obj = ''
		if i<9:
			mypath_obj = mypath+'/0'+str(i+1)
		else:
			mypath_obj = mypath+'/'+str(i+1)
		not_detect = 0
		for j in tqdm(range(len(onlyfiles_png[i]))):
			
			img_name = onlyfiles_png[i][j]
			txt_name = onlyfiles_txt[i][j]
			
			img_name_split = img_name.split('_')
			txt_name_split = txt_name.split('_')

			if img_name_split[1] != txt_name_split[1]:
				print('Mismatched!')
				sys.exit()


			pose_path = mypath_obj+'/'+txt_name
			# Load pose in degrees
			pose_annot = open(pose_path, 'r')
			R = []
			for line in pose_annot:
				line = line.strip('\n').split(' ')
				L = []
				if line[0] != '':
					for nb in line:
						if nb == '':
							continue
						L.append(float(nb))
					R.append(L)

			R = np.array(R)
			T = R[3,:]
			R = R[:3,:]
			pose_annot.close()

			R = np.transpose(R)

			roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
			yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
			pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi

			if abs(roll) > 99 or abs(yaw) > 99 or abs(pitch) > 99:
				not_detect += 1
				continue


			img = cv2.imread(mypath_obj+'/'+img_name)
			
			img_h = img.shape[0]
			img_w = img.shape[1]
			img = img[:, img_w//4:,:]
			img_h = img.shape[0]
			img_w = img.shape[1]

			if j==0:
				[xw1_pre,xw2_pre,yw1_pre,yw2_pre] = [0,0,0,0]
			detected = detector.detect_faces(img)
			
			if len(detected) > 0:
				dis_list = []
				XY = []
				area_list = []
				for i_d, d in enumerate(detected):
					
					xv = []
					yv = []
					for key, value in d['keypoints'].items():
						xv.append(value[0]) 
						yv.append(value[1])
					
					if d['confidence'] > 0.95:
						x1,y1,w,h = d['box']
						x2 = x1 + w
						y2 = y1 + h
					
						
						h = y2-y1
						w = x2-x1
						print("Face area: ", h*w, img_h, img_w)
						if h*w < 3500:
							not_detect += 1
							continue

					
						diff = abs(h-w) 
					
						if h > w: #If the detection box is hight
							if (x2 + diff//4 < img_w - 1) and (x1 - diff//4> 0) and (y2 - diff//4 > 0) and (y1 + diff//4 < img_h-1):   
								xw2 = x2 + diff//4
								xw1 = x1 - diff//4
								yw2 = y2 - diff//4
								yw1 = y1 + diff//4  
							elif (x2 + diff//2 < img_w-1) and x1 - diff//2 > 0:
								xw2 = x2 + diff//2
								xw1 = x1 - diff//2
								yw1 = y1 
								yw2 = y2 
							elif y2 - diff//2 > 0 and y1 + diff//2 < img_h -1:
								yw2 = y2 - diff//2
								yw1 = y1 + diff//2   
								xw1 = x1 
								xw2 = x2
							
			 

						else:
							if (y2 + diff//4 < img_h-1) and (y1 -  diff//4 > 0) and (x2- diff//4 > 0) and (x1+ diff//4 < img_w-1): 
								yw2 = y2 + diff//4
								yw1  = y1 -  diff//4
								xw2 = x2- diff//4
								xw1 = x1+ diff//4
							elif y2 + diff//2 < img_h-1 and y1 - diff//2 > 0:
								yw2 = y2 + diff//2
								yw1  = y1 -  diff//2
								xw1 = x1 
								xw2 = x2
							elif x2 - diff//2 > 0 and x1 + diff//2 < img_w -1:
								xw2 = x2- diff//2
								xw1 = x1+ diff//2
								yw1 = y1 
								yw2 = y2
							

						h = yw2-yw1
						w = xw2-xw1
						xw1 = max(int(xw1 - ad * w), 0)
						yw1 = max(int(yw1 - ad * w), 0)
						xw2 = min(int(xw2 + ad * h), img_w - 1)
						yw2 = min(int(yw2 + ad * h), img_h - 1)

						
					
						

						XY.append([xw1,xw2,yw1,yw2])
						cx = (xw1 +xw2)//2
						cy = (yw1 +yw2)//2
						dis_betw_cen = np.sqrt((cx-img_w/2)**2+(cy-img_h/2)**2)
						# dis_betw_cen = np.abs(xw1-img_w*2/3)+np.abs(yw1-img_h*2/3)
						dis_list.append(dis_betw_cen)
			
						area_list.append(h*w)
				
				if len(dis_list)>0:
					min_id = np.argmin(dis_list)
					[xw1,xw2,yw1,yw2] = XY[min_id]
					not_detect = 0
				else:
					if randFlag[i] == 0: not_detect += 1; continue
					if not_detect > 25: continue
					not_detect += 1
					img = cv2.resize(img[yw1_pre:yw2_pre + 1, xw1_pre:xw2_pre + 1, :], (img_size, img_size))
					if isPlot:
						print('No face detected! Use previous frame detected location.')
					
					# Checking the cropped image
					if isPlot:
						cv2.imshow('check',img)
						k=cv2.waitKey(10)
					img = cv2.resize(img, (img_size, img_size))
					cont_labels = np.array([yaw, pitch, roll])
					if randFlag[i] == 1:
						out_imgs_train.append(img)
						out_poses_train.append(cont_labels)
					continue

				cx = (xw1+xw2)//2
				cy = (yw1+yw2)//2
				cx_prev = (xw1_pre+xw2_pre)//2
				cy_prev = (yw1_pre+yw2_pre)//2
				dis_betw_frames = np.sqrt((cx-cx_prev/2)**2+(cy-cy_prev/2)**2)
				
				
				# dis_betw_frames = np.abs(xw1-xw1_pre)
				if dis_betw_frames < 250 or j==0:
					not_detect = 0
					img = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
					[xw1_pre,xw2_pre,yw1_pre,yw2_pre] = [xw1,xw2,yw1,yw2]
					if isPlot:
						print([xw1_pre,xw2_pre,yw1_pre,yw2_pre])
						cv2.imshow('check',img)
						k=cv2.waitKey(10)
					img = cv2.resize(img, (img_size, img_size))
					cont_labels = np.array([yaw, pitch, roll])
					
					if randFlag[i] == 1:
						out_imgs_train.append(img)
						out_poses_train.append(cont_labels)
					
					elif randFlag[i] == 0:
						
						out_imgs_test.append(img)
						out_poses_test.append(cont_labels)
				else:

					max_id = np.argmax(area_list)
					[xw1,xw2,yw1,yw2] = XY[max_id]
					
					img = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
					[xw1_pre,xw2_pre,yw1_pre,yw2_pre] = [xw1,xw2,yw1,yw2]
					# Checking the cropped image
					if isPlot:
						print([xw1_pre,xw2_pre,yw1_pre,yw2_pre])
						print('Distance between two frames too large! Use previous frame detected location.')
					
						cv2.imshow('check',img)
						k=cv2.waitKey(10)
					img = cv2.resize(img, (img_size, img_size))
					cont_labels = np.array([yaw, pitch, roll])
					if randFlag[i] == 1:
						out_imgs_train.append(img)
						out_poses_train.append(cont_labels)
					# elif randFlag[i] == 0:
					# 	out_imgs_test.append(img)
					# 	out_poses_test.append(cont_labels)
			else:
				# pass
				if not_detect > 25: 
					continue
				not_detect += 1
				img = cv2.resize(img[yw1_pre:yw2_pre + 1, xw1_pre:xw2_pre + 1, :], (img_size, img_size))
				
				if isPlot:
					print('No face detected! Use previous frame detected location.')
				
				# Checking the cropped image
				if isPlot:
					cv2.imshow('check',img)
					k=cv2.waitKey(10)
				img = cv2.resize(img, (img_size, img_size))
				cont_labels = np.array([yaw, pitch, roll])
				if randFlag[i] == 1:
					out_imgs_train.append(img)
					out_poses_train.append(cont_labels)
			# 	# elif randFlag[i] == 0:
			# 	# 	out_imgs_test.append(img)
			# 	# 	out_poses_test.append(cont_labels)
	np.savez(output_train_path,image=np.array(out_imgs_train), pose=np.array(out_poses_train), img_size=img_size)
	np.savez(output_test_path,image=np.array(out_imgs_test), pose=np.array(out_poses_test), img_size=img_size)


if __name__ == '__main__':
	main()