from PIL import Image
import matplotlib.pylab as plt
import h5py
import numpy as np
import os

def file_create(file_name,height, width):
	num = {}
	hf = h5py.File('Data/'+ file_name +'/data.h5','w')
	PATH = os.path.relpath(os.path.join('Data',file_name))	
	list_modes = ['train','test'] 
	list_classes = ['NORMAL','PNEUMONIA']
	# list_classes = ['cat','non-cat']
	for mode in list_modes:		
		Images = []
		Labels = []

		for class_type in list_classes:
			directory = PATH + '/' + mode + '/' + class_type +'/'			
			for image_file in os.listdir(directory):				
				img_cur = Image.open(directory + image_file)			
				img_cur = np.asarray(img_cur.resize((height,width)))
				Images.append(img_cur)
				Labels.append(int(class_type=='cat'))					

		Images = np.array(Images)
		Labels = np.array(Labels).reshape(1,len(Labels))
		
		hf.create_dataset(name='X_'+ mode ,  dtype ='i', data=Images)
		hf.create_dataset(name='Y_'+ mode ,  dtype ='i', data=Labels)

	hf.close()	

def load_data(file_name):
	height = 128
	width = 128
	# channels = 2
	print('loading data......')
	num = file_create(file_name,height, width)
	
	hf = h5py.File('Data/'+ file_name +'/data.h5','r')

	X_train = hf.get('X_train')
	Y_train = hf.get('Y_train')
	X_test = hf.get('X_test')
	Y_test = hf.get('Y_test')

	print('train samples: ',X_train.shape[0])	
	print('test samples: ',X_test.shape[0])

	hf.close()
	return X_train, Y_train, X_test, Y_test

load_data('chest_xray')