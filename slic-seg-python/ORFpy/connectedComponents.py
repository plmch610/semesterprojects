import numpy as np
import math
from orf import ORF
from utils import dataRange
from firstSliceDataset import dataset_ignit
from PIL import Image
import cv2
from tqdm import tqdm
import datasetManagement


def ConnectedComponents(preds, img_cl, connection_map, x_detection, y_detection):
    connectivity = 8
    output = cv2.connectedComponentsWithStats(preds, connectivity, cv2.CV_32S)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]

    target_labelnum_detector = np.multiply(labels, connection_map)
    label_number = target_labelnum_detector[x_detection][y_detection]/255
    final_segmentation = np.copy(labels)
    newdatas = []
    count = 0
    for i in range(len(final_segmentation)):
        for j in range(len(final_segmentation[0])):
            if final_segmentation[i][j] != label_number:
                final_segmentation[i][j]=0
                if img_cl[i][j][0] != 0:
                    newdatas.append([img_cl[i][j], 0])
                elif img_cl[i][j][0] == 0 and count%20 == 0:
                    count+=1
                    newdatas.append([img_cl[i][j], 0])
            else:
                final_segmentation[i][j]=255
                newdatas.append([img_cl[i][j], 1])


    return(final_segmentation, newdatas)