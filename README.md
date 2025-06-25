This repo contains the training and inference code for ostracod's motile reponse classification. Motile response is actively swimming vs drifting behaviour

__Dataset Description:__
- We have curated a novel dataset of ostracods hatched in a controlled labtrory environment 
- We perform imaging with a custom made structure(imaging chamber) where the water sample consisting ostracods and debris are passed. In the chamber and water samples are stirred to simulate movement and videos are captured
- From the captured videos we track all the orgnisms, our modeling starts with the tracking data. 
- The tracking data is written in a .txt file where object id: [(frame1, x1-coordinate, y1-coordinate)...(framen, xn-coordinate, yn-coordinate)] are written we perform training of the statistical models.
- __all_files__ folders contains 4 subfolders of the .txt files
__Data Preprocessing:__ 
- we modify the .txt file name with the date and time when imaging is performed. In some sample files, the ostracods and non-ostracods are seperated but it is not consistent for all the sample files. 
- we parse the .txt file and save the tracking data into a dictionary. We modify the object id also it is because the object id in one sample file doesn't indicate the same object id in the another sample file. 
- our modeling is based on displacements meaning dx=(x2-x1)/(frame2-frame1), dy=(y2-y1)/(frame2-frame1), therefore for n tracking data for one example organism we get n-1 displacements
- we apply automatic labeling (average-variance) on displacements for sepearating swimming and drifting organisms only for the .txt files where ostracods and non-ostracods aren't seperated.
  
 ##all these steps are done in the __driver_data_preprocessing.py__ file####
- we apply Z-score normalization to the displacements so that modeling can capture different water variablity.

##we compute the required statitics in the __driver_data_preprocessing.py__ file###
  
__Statistical Modeling:__
- we divide the imaging chamber into 5*5 times grid because inside the water chamber, water flow is not uniform. 
- we apply MGD on each grid cell and calcualte the probabilistic value for each displacements for an object.

##__driver_GridDisplacementModel.py__ identifies the grid cell location, computes grid statisitics, applies normalization and computes the final model parameters###

- we classify the motile response with two types of models: 1)__Outlier Model:__ where grid MGDS are trained using only drifting displacements and actively swimming organisms are classified as outlier
                                                            2)__Bayesian Model:__ where grid MGDs are trained using both the drifting and actively swimming displacements and their log sum values are compared to assign class.
  
##__GridOutlierModel.py__ computes threshold for the outlier model and classifies each object. 

##__GridBayesianModel.py__ classifies each object under each grid probabilisitic values. Furthermore, it adjusts the margin for the Bayesian Model. 

##__driver_main.py__ asks user choice, based on choice the program run modeling training and infers with the toxic induced objects. 

##__driver_preprocessing_util.py__ runs the model training and inference tasks. 

##__visualize_object_trajectory.py__ contains all the visualization related functions.  
