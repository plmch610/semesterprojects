import numpy as np
import math
from orf import ORF
from utils import dataRange
from firstSliceDataset import dataset_ignit
#import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
import datasetManagement
from connectedComponents import ConnectedComponents



def firstSliceSegmentation(file_scribbled, file_clean):
    print("-------- Start first Slice Segmentation --------")
    datasetManagement.ignitTrainData()
    train_data, test_data, scribbles_position = dataset_ignit(file_scribbled, file_clean)
    X = train_data[:, 0]
    y = train_data[:, 1]


    scribbles_map = np.zeros((448, 448))
    red_scribble_map = np.zeros((448, 448))
    blue_scribble_map = np.zeros((448, 448))
    for k in range(len(scribbles_position[0])):
        pos = scribbles_position[0][k]
        xi, yi = pos[0], pos[1]
        scribbles_map[xi][yi] = 127
        blue_scribble_map[xi][yi] = 255

    for k in range(len(scribbles_position[1])):
        pos = scribbles_position[1][k]
        xi, yi = pos[0], pos[1]
        scribbles_map[xi][yi] = 255
        red_scribble_map[xi][yi] = 255

    scribbles_map = np.array(scribbles_map)
    blue_scribble_map = np.array(blue_scribble_map)
    red_scribble_map = np.array(red_scribble_map)

    # setting parameters for ORF. For more details: >>> help(ORF).
    param = {'minSamples': 100, 'minGain': .01, 'numClasses': 2, 'xrng': dataRange(X), 'maxDepth': 10}
    orf = ORF(param,numTrees=20)
    for i in tqdm(range(len(X))):
        orf.update(X[i],y[i])



    img_cl = cv2.imread(file_clean)
    img_scribbled = cv2.imread(file_scribbled)
    xtest = test_data
    preds = np.array(orf.predicts(xtest))
    preds = preds.reshape(448, 448)
    preds = np.uint8(preds*255)

    x_detection = scribbles_position[1][0][0]
    y_detection = scribbles_position[1][0][1]

    final_segmentation, newdatas = ConnectedComponents(preds, img_cl, red_scribble_map, x_detection, y_detection)

    datasetManagement.deleteTrainData()
    datasetManagement.addDatas(newdatas)
    datasetManagement.numberTrainingData()
    print("-------- End first Slice Segmentation --------")

    return(final_segmentation, orf)






