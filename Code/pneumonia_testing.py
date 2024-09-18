import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
from PIL import Image, ImageTk
import cv2


window = tk.Tk()

window.title("Pneumonia Detection")

window.geometry("600x600")
window.configure(background ="grey")

title = tk.Label(text="  Click Select Image button to choose an image from test directory  ", background = "white", fg="Brown", font=("", 15))
title.grid()

def pneumonia():
    
    rem = "The Image provided has Pneumonia"
    remedies = tk.Label(text=rem, background="white",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = "Please consult a professional for futher assistance"
    remedies1 = tk.Label(text=rem1, background="white",
                        fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    answer = "Pneumonia"


def normal():
    
    rem = "The Image provided is Normal "
    remedies = tk.Label(text=rem, background="white",
                          fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = "Please consult a professional for futher assistance"
    remedies1 = tk.Label(text=rem1, background="white",
                             fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    answer = "Normal"



def analysis():
    import cv2 
    import numpy as np  
    import os  
    from random import shuffle  
    from tqdm import tqdm  
    
    verify_dir = 'test'
    print("path " + verify_dir)
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'pneumoniadetection-{}-{}.model'.format(LR, '2conv-basic')

    def process_verify_data():
        verifying_data = []
        for img in tqdm(os.listdir(verify_dir)):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
        np.save('verify_data.npy', verifying_data)
        return verifying_data

    verify_data = process_verify_data()

    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    import tensorflow as tf
    tf.compat.v1.reset_default_graph()

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

    import matplotlib.pyplot as plt

    fig = plt.figure()

    for num, data in enumerate(verify_data):

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        model_out = model.predict([data])[0]
        print(model_out)
        print('model {}'.format(np.argmax(model_out)))
 

        if np.argmax(model_out) == 0:
            str_label = 'normal'
        elif np.argmax(model_out) == 1:
            str_label = 'pneumonia'


        if str_label == 'normal':
            status= 'Normal'

            message = tk.Label(text='Status: '+status, background="white",
                           fg="green", font=("", 15))
            message.grid(column=0, row=4, padx=10, pady=10)
            normal ()

        elif str_label == 'pneumonia':
            diseasename = "Pneumonia"
            disease = tk.Label(text='Status: ' + diseasename, background="white",
                               fg="green", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)

            pneumonia ()

       
def openphoto():
    dirPath = "test"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)
        
    fileName = askopenfilename(initialdir='C://Users//ravis//Documents//ai//test', title='Select image for analysis ',
                           filetypes=[('Sample Images', '.JPEG')])
    dst = "test"
    print(fileName)
    print (os.path.split(fileName)[-1])
    if os.path.split(fileName)[-1].split('.') == 'h (1)':
        print('dfdffffffffffffff')
    shutil.copy(fileName, dst)
    load = Image.open(fileName)
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="300", width="575")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=0, row=1, padx=10, pady = 10)
    title.destroy()
    button1 = tk.Button(text="Select Image", command = openphoto)
    button1.grid(column=0, row=3, padx=10, pady = 10)
    button2 = tk.Button(text="Analyse Image", command=analysis)
    button2.grid(column=0, row=2, padx=10, pady = 10)

button1 = tk.Button(text="Select Image", command = openphoto)
button1.grid(column=0, row=1, padx=10, pady = 10)

window.mainloop()



