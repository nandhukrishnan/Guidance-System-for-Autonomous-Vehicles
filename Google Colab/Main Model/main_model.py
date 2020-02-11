from google.colab import drive
drive.mount('/content/drive')

!unzip "/content/drive/My Drive/M.Tech Project/Main Model/driving_dataset.zip"

import numpy as np
import pandas as pd
import os, shutil, cv2, random
from keras.models import Model
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D,BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense,Input,ELU
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from scipy import pi
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.regularizers import l2
from keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

xs = []
ys = []
original_data_path = r"/content/driving_dataset"
#original_data_path = r"/content/drive/My Drive/M.Tech Project/Main Model/driving_dataset"

with open(original_data_path +"/data.txt") as f:
    for line in f:
        xs.append(original_data_path+"/" + line.split()[0])
        ys.append(float(line.split()[1]) * pi / 180)

num_images = len(xs)
val_start_point = int(len(xs) * 0.6)
val_end_point = int(len(xs) *0.8)
                      
train_xs = xs[:val_start_point]
train_ys = ys[:val_start_point]
val_xs = xs[val_start_point : val_end_point]
val_ys = ys[val_start_point : val_end_point]
test_xs = xs[val_end_point:]
test_ys = ys[val_end_point:]
print("Total Number of Data-set: "+str(num_images))
print("Total Number of Train Data-set: "+str(len(train_xs)))
print("Total Number of Validation Data-set: "+str(len(val_xs)))
print("Total Number of Test Data-set: "+str(len(test_xs)))

read_image_height=200
read_image_width=200

image_height=100
image_width=200
channels = 3

train_dataset = np.ndarray(shape=(len(train_xs), image_height, image_width, channels), dtype=np.float32)
val_dataset = np.ndarray(shape=(len(val_xs), image_height, image_width, channels), dtype=np.float32)
test_dataset = np.ndarray(shape=(len(test_xs), image_height, image_width, channels), dtype=np.float32)

i = 0
for file in train_xs:
    img = load_img(file, target_size=(read_image_height, read_image_width)) 
    x = img_to_array(img)  
    x *= 1./255
    train_dataset[i] = x[75:175,:,:]
    del(x)
    i += 1
    if i % 10000 == 0:
        print("%d images to array" % i)
print("All Train images to array!")


i = 0
for file in val_xs:
    img = load_img(file, target_size=(read_image_height, read_image_width)) 
    x = img_to_array(img)  
    x *= 1./255
    val_dataset[i] = x[75:175,:,:]
    del(x)
    i += 1
    if i % 10000 == 0:
        print("%d images to array" % i)
print("All validation images to array!")        


i = 0
for file in test_xs:
    img = load_img(file, target_size=(read_image_height, read_image_width)) 
    x = img_to_array(img)  
    x *= 1./255
    test_dataset[i] = x[75:175,:,:]
    del(x)
    i += 1
    if i % 10000 == 0:
        print("%d images to array" % i)
        
print("All Test images to array!")

image_height=100
image_width=200
channels = 3

input_size = (image_height,image_width,channels)
act_funct = "relu"
def create_model():
  model = Sequential()
  model.add(Conv2D(12, (1, 1), padding="same", activation=act_funct,input_shape=input_size)) #inception layer
  model.add(Conv2D(24, (5, 5),strides=(3, 3), padding="same", activation=act_funct,use_bias=True,
            kernel_initializer='he_uniform', bias_initializer='he_uniform', 
            kernel_regularizer=l2(1.e-4), bias_regularizer=l2(1.e-4)))
  model.add(Conv2D(18, (1, 1), padding="same", activation=act_funct))
  model.add(Conv2D(36, (5, 5),strides=(3, 3), activation=act_funct,use_bias=True,
            kernel_initializer='he_uniform', bias_initializer='he_uniform', 
            kernel_regularizer=l2(1.e-4), bias_regularizer=l2(1.e-4)))
  model.add(Conv2D(18, (1, 1),strides=(1, 1), padding="same", activation=act_funct))
  model.add(Conv2D(48, (5, 5),strides=(3, 3), padding="same", activation=act_funct,use_bias=True,
            kernel_initializer='he_uniform', bias_initializer='he_uniform', 
            kernel_regularizer=l2(1.e-4), bias_regularizer=l2(1.e-4)))
  model.add(Conv2D(18, (1, 1),padding="same", activation=act_funct))
  model.add(Conv2D(64, (3, 3),strides=(1, 1), padding="same", activation=act_funct,use_bias=True,
            kernel_initializer='he_uniform', bias_initializer='he_uniform', 
            kernel_regularizer=l2(1.e-4), bias_regularizer=l2(1.e-4)))
  model.add(Conv2D(18, (1, 1),padding="same", activation=act_funct))
  model.add(Conv2D(64, (3, 3),strides=(1, 1), padding="same", activation=act_funct,use_bias=True,
            kernel_initializer='he_uniform', bias_initializer='he_uniform', 
            kernel_regularizer=l2(1.e-4), bias_regularizer=l2(1.e-4)))
  model.add(Flatten())
  model.add(Dropout(0.2))
  model.add(Dense(1164, activation=act_funct,use_bias=True,
            kernel_initializer='he_uniform', bias_initializer='he_uniform', 
            kernel_regularizer=l2(1.e-4), bias_regularizer=l2(1.e-4)))
  model.add(Dropout(0.2))
  model.add(Dense(100, activation=act_funct,use_bias=True,
            kernel_initializer='he_uniform', bias_initializer='he_uniform', 
            kernel_regularizer=l2(1.e-4), bias_regularizer=l2(1.e-4)))
  model.add(Dropout(0.2))
  model.add(Dense(50, activation=act_funct,use_bias=True,
            kernel_initializer='he_uniform', bias_initializer='he_uniform', 
            kernel_regularizer=l2(1.e-4), bias_regularizer=l2(1.e-4)))
  model.add(Dropout(0.2))
  model.add(Dense(10, activation=act_funct,use_bias=True,
            kernel_initializer='he_uniform', bias_initializer='he_uniform', 
            kernel_regularizer=l2(1.e-4), bias_regularizer=l2(1.e-4)))
  #model.add(Dropout(0.5))
  model.add(Dense(1))
  return model

model = create_model()

model.summary()

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.001, decay=0.0, amsgrad=False)
model.compile(loss='mse',optimizer=adam)

bat_size=256
t_steps_per_e=int(len(train_dataset)/bat_size)
v_steps_per_e=int(len(val_dataset)/bat_size)
ep=50

print("Data Augumantation started")
train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    width_shift_range=0.2,
    height_shift_range=0.2)
train_datagen.fit(train_dataset)
train_generator = train_datagen.flow(train_dataset,train_ys,batch_size=bat_size)

val_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    width_shift_range=0.2,
    height_shift_range=0.2)
val_datagen.fit(val_dataset)
val_generator = train_datagen.flow(val_dataset,val_ys,batch_size=bat_size)

test_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    width_shift_range=0.2,
    height_shift_range=0.2)
test_datagen.fit(test_dataset)
test_generator = test_datagen.flow(test_dataset,test_ys,batch_size=bat_size)

import keras
tb = keras.callbacks.TensorBoard(log_dir='logs')
checkpointer = keras.callbacks.ModelCheckpoint(filepath='/content/drive/My Drive/M.Tech Project/Main Model/base_model_weights.hdf5', verbose=1)

#history = model.fit_generator(train_generator,steps_per_epoch=t_steps_per_e,epochs=ep,validation_data=val_generator,validation_steps=v_steps_per_e,callbacks=[tb,checkpointer])
history = model.fit(x=train_dataset, y=train_ys, batch_size=64, epochs=100, verbose=1, validation_data=(val_dataset,val_ys),initial_epoch=30)
model.save('/content/drive/My Drive/M.Tech Project/Main Model/main_model.h5')

model = load_model("/content/drive/My Drive/M.Tech Project/Main Model/main_model.h5")

print(model.evaluate(test_dataset,test_ys))
Predicted_test = model.predict(test_dataset)

Predicted_test[:,0].shape
len(test_ys)

def display_diff(y_act,y_pre):
  z = y_act/y_pre
  plt.hist(z, label='Training DIffernce')
  plt.title('Training and validation loss')
  plt.legend()
  plt.show()
  
display_diff(test_ys,Predicted_test[:,0])

print(model.evaluate_generator(test_generator))
Predicted_test = model.predict_generator(test_generator)

i=0
for i in range(0,10):
  z=random.randint(0,9081)
  plt.imshow(test_dataset[z])
  plt.show()
  print(test_ys[z])
  i+=1

