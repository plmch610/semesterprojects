import numpy as np
import math
from orf import ORF
from utils import dataRange
from firstSliceDataset import dataset_ignit
from PIL import Image
import cv2
from tqdm import tqdm
import datasetManagement
import os
import shutil
import glob
from sliceDataset import sliceDataset
from connectedComponents import ConnectedComponents
from statistics import mean


def propagation(folder, firstSlice, orf, FirstSliceSegmentation):
    print("-------- Start Propagation --------")
    slices = glob.glob(os.path.join(folder, "*.bmp"))

    indexFirstSlice = slices.index(firstSlice)
    slices.pop(indexFirstSlice)
    slices.sort()

    PropagationSegmentation = [FirstSliceSegmentation]
 
    #here only propagate over 3 slices to make all the implementation required (e.g. CRF)
    #change range(0,3) to range(len(slices))
    for k in range(0,3):
        print("-------- Start Slice"+ str(k+1) +" Propagation --------")
        currentSlice = slices[k]

        train_data, test_data, img_cl = sliceDataset(currentSlice)
        
        X = train_data[:, 0]
        y = train_data[:, 1]
        xtest = test_data
        param = {'minSamples': 100, 'minGain': .01, 'numClasses': 2, 'xrng': dataRange(X), 'maxDepth': 10}
        for i in tqdm(range(len(X))):
            orf.update(X[i],y[i])

        
        preds = np.array(orf.predicts(xtest))
        preds = preds.reshape(448,448)
        preds = np.uint8(preds*255)

        kernel = np.ones((3, 3), np.uint8)
        preds = cv2.morphologyEx(np.uint8(preds), cv2.MORPH_CLOSE, kernel)
        
        connection_map = PropagationSegmentation[-1]

        # Creating kernel
        
        connection_map = cv2.morphologyEx(np.uint8(connection_map), cv2.MORPH_CLOSE, kernel)

        ConnectionListx = []
        ConnectionListy = []
        for i in range(448):
            for j in range(448):
                if (connection_map[i][j] != 0) & (preds[i][j] != 0) :
                    x = i
                    y = j
                    ConnectionListx.append(x)
                    ConnectionListy.append(y)

        x = int(mean(ConnectionListx))
        y = int(mean(ConnectionListy))

        minimum = [((ConnectionListx[0]-x)**2+(ConnectionListy[0]-y)**2)**(0.5), ConnectionListx[0], ConnectionListy[0]]
        m = minimum[0]
        for i in range(len(ConnectionListx)):
            for j in range(len(ConnectionListy)):
                if minimum[0] > m :
                    m = ((ConnectionListx[i]-x)**2+(ConnectionListy[j]-y)**2)**(0.5)
                    minimum = [m, ConnectionListx[i], ConnectionListy[j]]

        x = minimum[1]
        y = minimum[2]

        final_segmentation, newdatas = ConnectedComponents(preds, img_cl, connection_map, x, y)

        #to cancel dataset update, comment these 3 lines
        datasetManagement.deleteTrainData()
        datasetManagement.addDatas(newdatas)
        datasetManagement.numberTrainingData()
     
        PropagationSegmentation.append(final_segmentation)
        print("-------- End Slice"+ str(k+1) +" Propagation --------")
        

        
    print("-------- End Propagation --------")
    return(PropagationSegmentation)
    
