from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import ResNet101 #for test
from tensorflow.keras.optimizers import Nadam #for test
import pandas as pd


def load_test(path):
    datagen_test = ImageDataGenerator(rescale=(1.0/255.0), validation_split=0.2)

    test_datagen_flow = datagen_test.flow_from_dataframe(
        dataframe=pd.read_csv(path + 'labels.csv'),
        directory='/datasets/faces/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=16,
        class_mode='raw',
        subset='validation',
        seed=12345)
    return test_datagen_flow


def load_train(path):
    datagen_train = ImageDataGenerator(rescale=(1.0/255.0), 
                                       validation_split=0.2, 
                                       rotation_range=15,
                                       width_shift_range=0.2, 
                                       height_shift_range=0.2)

    train_datagen_flow = datagen_train.flow_from_dataframe(
        dataframe=pd.read_csv(path + 'labels.csv'),
        directory='/datasets/faces/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=16,
        class_mode='raw',
        subset='training',
        seed=12345)
    return train_datagen_flow

def create_model(input_shape):
    optimizer = Adam(learning_rate = 0.0001)
    backbone = ResNet101(input_shape=input_shape, 
                        weights='imagenet', 
                        include_top=False)
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=1, activation='relu'))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def train_model(model, 
                train_datagen_flow, 
                val_datagen_flow, 
                batch_size=None, 
                epochs=30,
                steps_per_epoch=None, 
                validation_steps=None):

    if steps_per_epoch == None:
        steps_per_epoch = len(train_datagen_flow)
    if validation_steps == None:
        validation_steps = len(val_datagen_flow)

    model.fit(train_datagen_flow,  
              validation_data=val_datagen_flow,
              steps_per_epoch=steps_per_epoch, 
              epochs=epochs,
              validation_steps=validation_steps,
              verbose=2, 
              shuffle=True)
    return model