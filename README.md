This repo contains the training and inference code for ostracod's motile reponse classification. Motile response is actively swimming vs drifting behaviour
Dataset Description:
- We have curated a novel dataset of ostracods hatched in a controlled labtrory environment 
- We perform imaging with a custom made structure(imaging chamber) where the water sample consisting ostracods and debris are passed. In the chamber and water samples are stirred to simulate movement and videos are captured
- From the captured videos we track all the orgnisms, our modeling starts with the tracking data. 
- The tracking data is written in a .txt file where object id: [(frame1, x1-coordinate, y1-coordinate)...(framen, xn-coordinate, yn-coordinate)] are written we perform training of the statistical models.
Data Preprocessing: 
- we modify the .txt file name with the date and time when imaging is performed. In some sample files, the ostracods and non-ostracods are seperated but it is not consistent for all the sample files. 
- we parse the .txt file and save the tracking data into a dictionary. We modify the object id also it is because the object id in one sample file doesn't indicate the same object id in the another sample file. 
- our modeling is based on displacements meaning dx=(x2-x1)/(frame2-frame1), dy=(y2-y1)/(frame2-frame1), therefore for n tracking data for one example organism we get n-1 displacements
- we apply automatic labeling (average-variance) on displacements for sepearating swimming and drifting organisms only for the .txt files where ostracods and non-ostracods aren't seperated. 
- we apply Z-score normalization to the displacements so that modeling can capture different water variablity. 
Statistical Modeling:
- we divide the imaging chamber into 5*5 times grid because inside the water chamber, water flow is not uniform 
- we apply MGD on each grid cell and calcualte the probabilistic value for each displacements for an object
- we classify the motile response with two types of models: 1)Outlier Model: where grid MGDS are trained using only drifting displacements and actively swimming organisms are classified as outlier
                                                            2)Bayesian Model: where grid MGDs are trained using both the drifting and actively swimming displacements and their log sum values are compared to assign class. 
