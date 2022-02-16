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


train_data = datasetManagement.getData()

print(train_data)