import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import datasetManagement

def dataset_ignit(img_scribbled, img_clean):
    img_scrib = cv2.imread(img_scribbled)
    img_cl = cv2.imread(img_clean)

    blue, red = 0, 0
    blue_pixels = []
    red_pixels = []
    for i in range(len(img_scrib)):
        row = img_scrib[i]
        for j in range(len(row)):
            pix = row[j]
            #below equalization for blue scribbled pixel
            if (pix[0] == 255 and pix[1]==51 and pix[2]==4): 
                blue = blue + 1
                blue_pixels.append([i, j])

            #below equalization for red scribbled pixel
            if (pix[0] == 0 and pix[1]==38 and pix[2]==255): 
                red = red + 1
                red_pixels.append([i, j])

    train_data = []
    test_data = []
    scribbles_position = [[], []]
    for k in range(len(blue_pixels)):
        current_pix = blue_pixels[k]
        pixels = img_cl[current_pix[0]][current_pix[1]]
        train_data.append([pixels, 0])
        scribbles_position[0].append(current_pix)

    for k in range(len(red_pixels)):
        current_pix = red_pixels[k]
        pixels = img_cl[current_pix[0]][current_pix[1]]
        train_data.append([pixels, 1])
        scribbles_position[1].append(current_pix)

    for k in range(len(img_cl)):
        row = img_cl[k]
        for j in range(len(row)):
            test_data.append(img_cl[k][j])

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    np.random.shuffle(train_data)
    datasetManagement.addDatas(train_data)

    return(train_data, test_data, scribbles_position)

    




