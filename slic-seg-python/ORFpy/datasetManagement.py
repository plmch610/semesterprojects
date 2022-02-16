import numpy as np
import math
from orf import ORF
from utils import dataRange
from PIL import Image
import cv2
from tqdm import tqdm
import os
import csv 
import pandas as pd

def ignitTrainData():
    with open('../train_data/train_data.csv', 'w') as f:
        print('train_data.csv created')
        f.close()

def deleteTrainData():
    os.remove('../train_data/train_data.csv')

def addDatas(new_data):
    with open('../train_data/train_data.csv', 'a') as f:
        writer = csv.writer(f)

        writer.writerows(new_data)
        numberTrainingData()
        f.close()

def addData(new_data):
    with open('../train_data/train_data.csv', 'a') as f:
        writer = csv.writer(f)

        writer.writerow(new_data)
        numberTrainingData()
        f.close()


def getData():
    with open('../train_data/train_data.csv') as f:
        file = csv.reader(f)
        training_data = list(file)
        for k in range(len(training_data)):
            data = training_data[k][0]
            result = training_data[k][1]
            training_data[k][1] = int(result)

            data = list(map(int, data.split('[')[1].split(']')[0].split(' ')))
            training_data[k][0] = data
        f.close()
    
    return(training_data)


def numberTrainingData():
    with open('../train_data/train_data.csv') as f:
        file = csv.reader(f)
        training_data = list(file)
        print('There are ' + str(len(training_data)) + ' training data')
        f.close()



