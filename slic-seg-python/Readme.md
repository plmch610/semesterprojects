## Slic-Seg Python
Container folder for Slic-Seg project in Python
Main contributor: Paul Michel

## ORF from github link https://github.com/luiarthur/ORFpy

The structure of the repository is based on the structure of the repository of luiarthur from the link below.
From this link we find the implementation of the On-line Random Forest from the paper of Amir Saffari.
The README (e.g. READMEORFpy.md)  of the repository of luiarthur explains well how the implementation work.
We just have to note that the files used are orf.py, ort.py and tree.py from the ORFpy directory.

## How to use this implementation (but not complete) of Slic-Seg python 

This implementation was used on a conda environment using python 2.7.18 and open-cv 4.2.0. 

**main.py** : regroup all the files to make them work together, to use the data -> create 2 folder, 1 with first slice scribbled and one without the first slice scribbled (example of such folder with direction test_clean and test_scribbled)
  
**firstSliceDataset.py** : initiate the dataset using the scribbles, for scribbles use value of the RGBs pixel in the file or modify them appropriately, create train and test set.  
**firstSliceSegmentation.py** : segment the first slice.
  
**sliceDataset.py** : create train and test data for each propagation slice.  
**propagation.py** : propagation file. 
  
**connectedComponents.py** : connected components file.   
**datasetManagement.py** : dataset management file. 

When the code execution is finished, open-cv imshow windows appear. To end the execution of the code click on a image and press a key
