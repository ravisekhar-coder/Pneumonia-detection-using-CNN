import cv2                 
import numpy as np         
import os                  
from random import shuffle 
from tqdm import tqdm



TRAIN_DIR = 'C://Users//ravis//Documents//ai//train'
TEST_DIR =  'C://Users//ravis//Documents//ai//test'

IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'pneumoniadetection-{}-{}.model'.format(LR, '2conv-basic')

def label_img(img):
    word_label = img[0]
    print(word_label)
  
    if word_label == 'n':
        print('normal')
        return [1,0]
    elif word_label == 'p':
        print('pneumonia')
        return [0,1]


def create_train_data():
    training_data = []

    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        print('##############')
        print(label)
        path = os.path.join(TRAIN_DIR,img)
 
        img = cv2.imread(path,cv2.IMREAD_COLOR)

        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        training_data.append([np.array(img),np.array(label)])
    
    shuffle(training_data)

    images = np.array([item[0] for item in training_data])
    labels = np.array([item[1] for item in training_data])

    # Save images and labels separately
    np.save('train_images.npy', images)
    np.save('train_labels.npy', labels)
    return training_data



def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    images = np.array([item[0] for item in testing_data])
    labels = np.array([item[1] for item in testing_data])

    # Save images and labels separately
    np.save('test_images.npy', images)
    np.save('test_labels.npy', labels)
    return testing_data

train_data = create_train_data()

process_data=process_test_data()


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression




convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 128, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
    

train = train_data[:-3372]
test = train_data[-3372:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = [i[1] for i in train]
print(X.shape)
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
test_y = [i[1] for i in test]
print(test_x.shape)

model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), show_metric=True, run_id=MODEL_NAME)


model.save(MODEL_NAME)











        
