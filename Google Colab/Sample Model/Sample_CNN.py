from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dropout

from google.colab import drive
drive.mount('/content/drive')

model=Sequential()
model.add(Convolution2D(24,3,3,input_shape=(256,256,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(36,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(48,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(3,activation='sigmoid'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory(
    directory='/content/drive/My Drive/Dataset/training_set',
    target_size=(256,256),
    batch_size=32,
    class_mode='categorical')

test_set=test_datagen.flow_from_directory(
    directory='/content/drive/My Drive/Dataset/test_set',
    target_size=(256,256),
    batch_size=32,
    class_mode='categorical')

filepath="/content/drive/My Drive/Dataset/epochs:{epoch:03d}-val_acc:{val_acc:.3f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit_generator(
    training_set,
    steps_per_epoch=10,
    epochs=25,
    validation_data=test_set,
    validation_steps=5)

model.save('/content/drive/My Drive/Dataset/model.h5')
