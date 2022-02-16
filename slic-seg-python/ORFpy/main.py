import numpy as np
import math
from orf import ORF
from utils import dataRange
from firstSliceDataset import dataset_ignit
#import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
from firstSliceSegmentation import firstSliceSegmentation
import datasetManagement
from propagation import propagation

file_scribbled = '../test_scribbled/1.bmp'
file_clean = '../test_clean/1.bmp'
Final_segmentations = []
FirstSliceSegmentation, orf = firstSliceSegmentation(file_scribbled, file_clean)

PropagationSegmentation = propagation('../test_clean', '../test_clean/1.bmp', orf, FirstSliceSegmentation)

for k in range(len(PropagationSegmentation)):
    cv2.imshow("Segmentation"+str(k), np.uint8(PropagationSegmentation[k]))


datasetManagement.deleteTrainData()

cv2.waitKey(0)
cv2.destroyAllWindows()