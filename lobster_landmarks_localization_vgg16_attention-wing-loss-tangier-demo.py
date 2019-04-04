#!/usr/bin/env python
# coding: utf-8

# In[4]:


import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,help="path to input video")
args = vars(ap.parse_args())


# In[ ]:





# In[1]:


import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import os

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D,MaxPooling2D,Flatten,Conv1D,Softmax
from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split


import json
import numpy as np
import pandas as pd
from sklearn import preprocessing
import keras
from keras.layers import Input,Dense,Lambda,RepeatVector,Dot
from keras.models import Model
import os
import numpy as np
from keras.preprocessing import image as image_p
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
import time
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import pickle
import matplotlib.image as mpimg
from keras.preprocessing import image
from  matplotlib import pyplot
from keras.layers.normalization import BatchNormalization
import cv2
import seaborn as sns
import random
from PIL import Image
from sklearn.utils import class_weight
from keras.layers import Reshape,merge,Concatenate,Add,Dropout
import keras.backend as K
import math
from keras.activations import softmax,tanh
import tensorflow as tf
from keras.applications.vgg16 import VGG16


#import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split


import json
import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.image as mpimg
from  matplotlib import pyplot
import cv2
import random
from PIL import Image
from sklearn.utils import class_weight
import numpy
import codecs

import xmltodict, json

import imgaug as ia
from imgaug import augmenters as iaa
import math
from keras.callbacks import ModelCheckpoint


# In[2]:


bgr_img = cv2.imread(args["input"])


# In[5]:


def process_image_keypoints(img,rectangle_points,bbox_coords):
    desired_size = 224

    old_size = img.shape

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    point_list = rectangle_points
    rects = []
    for a,b in point_list:
        a = a*ratio
        b = b*ratio
        rects.append([a+left,b+top])
    bbox_coordinates = []
    for a,b in bbox_coords:
        a = float(a)*ratio
        b = float(b)*ratio
        bbox_coordinates.append([a+left,b+top])
    return new_im,rects,bbox_coordinates
def process_image_keypoints_nobox(img,rectangle_points):
    desired_size = 224

    old_size = img.shape

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    point_list = rectangle_points
    rects = []
    for a,b in point_list:
        a = a*ratio
        b = b*ratio
        rects.append([a+left,b+top])
    return new_im,rects

def process_image_keypoints_coords(img,bbox_coords):
    desired_size = 224

    old_size = img.shape

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    bbox_coordinates = []
    for a,b in bbox_coords:
        a = float(a)*ratio
        b = float(b)*ratio
        bbox_coordinates.append([a+left,b+top])
    return new_im,bbox_coordinates
# !pip install keras
#num_labels = len(np.unique(labels))

def process_image_keypoints_nobox_util(img):
    desired_size = 224

    old_size = img.shape

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return new_im,[left,top,ratio]


# In[ ]:


from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# In[16]:



def attention_block(x,K,labels,w,h):
    H = Conv2D(K, kernel_size=(w, h), padding="same")(x)
    H = BatchNormalization()(H)
    spatial_softmax = Lambda(lambda x:softmax(H,axis=3))(H)
    O = Conv2D(labels*K, kernel_size=(w, h), padding="same")(x)
    O = BatchNormalization()(O)
    rH = Lambda(lambda x:keras.backend.repeat_elements(x, rep=labels, axis=3))(spatial_softmax)
    o = Lambda(lambda x: tf.einsum('bijk,bijk->bk',x[0],x[1]))([O,rH])
    tempH = Conv2D(K, kernel_size=(w, h), padding="same")(x)
    tempH = BatchNormalization()(tempH)
    gh = Lambda(lambda x: softmax(tanh(tf.einsum('bijk,bijk->bk',x[0],x[1])),axis=1))([tempH,spatial_softmax])
    rgh = Lambda(lambda x:keras.backend.repeat_elements(x, rep=labels, axis=1))(gh)
    of = Lambda(lambda x: tf.einsum('bk,bk->bk',x[0],x[1]))([rgh,o])
    new_shape = Reshape((labels,K))(of)
    out = Lambda(lambda x: keras.backend.sum(x,axis=2))(new_shape)
    #x = Conv2D(K, kernel_size=(w, h), padding="same")(x)
    #inter = Lambda(lambda x: Flatten()(keras.backend.mean(x,axis=3)))(x)
    inter = Conv2D(128, kernel_size=(w, h), padding="same")(x)
    inter = MaxPooling2D(pool_size=(2, 2))(inter)
    inter = BatchNormalization()(inter)
    inter = Conv2D(128, kernel_size=(w, h), padding="same")(inter)
    inter = MaxPooling2D(pool_size=(2, 2))(inter)
    inter = BatchNormalization()(inter)
    print(inter.shape)
    inter = Conv2D(1, kernel_size=(w, h), padding="same")(inter)
    inter = Flatten()(inter)
    weight = Dense(labels,activation='tanh')(inter)
    return out,weight
K = 10
labels = 12
w,h = 3,3

cnn_base = VGG16(input_shape=(224,224,3),include_top=False,weights='imagenet')

for layer in cnn_base.layers:
    if layer.name=='block5_conv3':
        break
    layer.trainable = False

x = cnn_base.get_layer('block2_conv2').output
final4,weight4 = attention_block(x,K,labels,w,h)
#final1 = Lambda(lambda x: tf.einsum('bk,bk->bk',x[0],x[1]))([final1,weight1])
    

x = cnn_base.get_layer('block3_conv3').output
final1,weight1 = attention_block(x,K,labels,w,h)
#final1 = Lambda(lambda x: tf.einsum('bk,bk->bk',x[0],x[1]))([final1,weight1])

x = cnn_base.get_layer('block4_conv3').output
final2,weight2 = attention_block(x,K,labels,w,h)

#final2 = Lambda(lambda x: tf.einsum('bk,bk->bk',x[0],x[1]))([final2,weight2])

x = cnn_base.get_layer('block5_pool').output
inter = Lambda(lambda x: Flatten()(x[:,:,:,0]))(x)
weight3 = Dense(labels,activation='tanh')(inter)

x = Conv2D(128, kernel_size=(3, 3), padding="same")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(128, kernel_size=(3, 3), padding="same")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(labels, kernel_size=(1, 1), padding="same")(x)
x = Reshape((labels,),name='localization_output')(x)
#final3 = Lambda(lambda x: tf.einsum('bk,bk->bk',x[0],x[1]))([x,weight3])

#bag_of_models = Add()([final1,final2,final3])
# weight_concat = Concatenate()([weight1,weight2,weight4,weight3])
# weight_concat = Softmax()(weight_concat)
# final = Concatenate()([final1,final2,final4,x])

# final3 = Lambda(lambda x: tf.einsum('bk,bk->bk',x[0],x[1]))([weight_concat,final])
# final3 = Dense(22)(final3)

weight_concat = Concatenate()([weight1,weight2,weight3])
weight_concat = Reshape((labels,3))(weight_concat)
weight_concat = Softmax(axis=2)(weight_concat)

final = Concatenate()([final1,final2,x])
final = Reshape((labels,3))(final)


sum_final = Lambda(lambda x: tf.einsum('bij,bij->bij',x[0],x[1]))([weight_concat,final])
sum_final = Lambda(lambda x: keras.backend.sum(x,axis=2))(sum_final)
sum_final = Dense(labels)(sum_final)

landmark_model = Model(inputs=cnn_base.input, outputs=sum_final)
sgd = keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
landmark_model.compile(loss="mean_squared_error", optimizer=sgd)
print(landmark_model.summary())

with open('lobster-demo/landmark-weightsv20.hd5','rb') as f:
    model_weights = pickle.load(f)
    print("loaded model")
landmark_model.set_weights(model_weights)


# In[19]:


#bgr_img = cv2.imread("/tf/data/tangier-visit2/DSC_0490.JPG")
bgr_img = cv2.imread(args["input"])
img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

image_temp,corr = process_image_keypoints_nobox_util(img)
img_temp = (image_temp[:,:,:] / 255.0).astype(np.float64)
tuples = []
pred_landmarks = landmark_model.predict(np.expand_dims(img_temp,0))[0]
#image_temp = (img_temp * 255.0).astype(np.uint8)
for i in range(6):
    tuples.append((pred_landmarks[i*2],pred_landmarks[i*2+1]))
for i in range(6):
    image_temp = cv2.circle(image_temp,(int(tuples[i][0]),int(tuples[i][1])), 3, (0,0,255), -1)
plt.figure(figsize=(15,15))
RGB_img = cv2.cvtColor(image_temp, cv2.COLOR_BGR2RGB)
print("saving cropped image")
cv2.imwrite("cropped_predictions_model3.jpg",RGB_img)


# In[21]:


[left,top,ratio] = corr
global_tuples = []
for i in range(6):
    global_tuples.append((int((tuples[i][0]-left)/ratio),int((tuples[i][1]-top)/ratio)))
for i in range(6):
    img = cv2.circle(img,(int(global_tuples[i][0]),int(global_tuples[i][1])), 40, (0,0,255), -1)
print("saving global image")
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite("predictions_model3.jpg",RGB_img)


# In[27]:





# In[36]:


carapace_start = np.array([(tuples[0][0] + tuples[1][0])/2.0,(tuples[0][1] + tuples[1][1])/2.0])
carapace_end = np.array([(tuples[2][0] + tuples[3][0])/2.0,(tuples[2][1] + tuples[3][1])/2.0])
tail_end = np.array([(tuples[4][0] + tuples[5][0])/2.0,(tuples[4][1] + tuples[5][1])/2.0])

aperture_length = 35
import numpy as np
carapace_length = np.linalg.norm(carapace_start-carapace_end)
tail_length = np.linalg.norm(carapace_end-tail_end)
distance_to_obj = 501.65
focal_length = 18

carapace_length_pixels = (carapace_length*distance_to_obj)/focal_length
tail_length_pixels = (tail_length*distance_to_obj)/focal_length

print("Carapace length in mm",carapace_length_pixels/aperture_length)
print("Tail length in mm",tail_length_pixels/aperture_length)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[24]:





# In[ ]:





# In[26]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




