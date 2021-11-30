#%env CUDA_VISIBLE_DEVICES=2

import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
#from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

from tkinter import Tk
from tkinter.filedialog import askopenfilename

# https://stackoverflow.com/questions/34588464/python-how-to-capture-image-from-webcam-on-click-using-opencv/34588758
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

import cv2
import serial
import operator



image_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

idx_to_class = {0 : 'bottles', 1 : 'cans', 2: 'waste'}

def predict(model, test_image_name):
    '''
    Function to predict the class of a single test image
    Parameters
        :param model: Model to test
        :param test_image_name: Test image

    '''
    
    transform = image_transforms['test']

    test_image = Image.open(test_image_name)
    plt.imshow(test_image)
    
    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
    
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(3, dim=1)
        scoreDict = {}  # index to score pairing
        for i in range(3):
            scoreDict[topclass.cpu().numpy()[0][i]] = topk.cpu().numpy()[0][i]
            print("Prediction", i+1, ":", idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ", topk.cpu().numpy()[0][i])
        return scoreDict

def wait_til_phrase(ser, phrase):
    while True:
        response = ser.read_until().decode("utf-8")
        print(list(response), end="")
        if response == phrase:
            print("\nPhrase found!")
            break

model = torch.load('data_model_29.pt')
cam = cv2.VideoCapture(0)

scores = {}


print("")
input("Enter to Start:\t")

with serial.Serial("COM6", 9800, timeout=1) as ser:
    for i in range(3):
        ser.write(b'a') # bah
        time.sleep(0.5)
    ser.write(b'W')

    while True:
        # wait for arduino signal to read
        wait_til_phrase(ser, "s\r\n")
        print("Arduino signal received!")
        print("")

        input("Enter to Continue:\t")
        print("")

        # do pytorch stuff
        print(idx_to_class)
        ret, frame = cam.read()
        cv2.imwrite("./cambuffer/bup.jpg", frame)
        scores = predict(model, "./cambuffer/bup.jpg")
        print(scores)
        print("")

        most_confident_idx = max(scores.items(), key=operator.itemgetter(1))[0]

        # send command to arduino
        if scores[0] > scores[1]:  # bottle detected
            print("BOTTLE!!!")
            ser.write(b'L')
        else:
            print("CAN!!!")
            ser.write(b'R')

        # that's it!
        
    ser.close()

input("Bye!")
