import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import datasetManagement

def sliceDataset(img_clean):
    img_cl = cv2.imread(img_clean)

    train_data = datasetManagement.getData()
    test_data = []

    for k in range(len(img_cl)):
        row = img_cl[k]
        for j in range(len(row)):
            test_data.append(img_cl[k][j])

    
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    np.random.shuffle(train_data)

    return(train_data, test_data, img_cl)
