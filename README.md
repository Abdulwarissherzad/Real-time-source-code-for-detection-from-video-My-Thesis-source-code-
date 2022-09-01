# Real-time-source-code-for-detection-from-video-My-Thesis-source-code-
The main idea behind this study is to detect and classify four types of road damage, and furthermore to enhance the real time detection capabilities; therefore, in this study we are going to apply YOLOv4 for road damage detection. After collection of the dataset  and annotation procedure using the LabelImg tool, the data is ready for training. We randomly divide the dataset into training and validation datasets in a 9:1 ratio. The training dataset is used to train the model and extract significant features, and a valid dataset can lead the model in the right direction. Validation is used to fit the model on the training set and tune model hyper parameters on each epoch in the training phase to validate the model. After finding the optimal weight, we can test the model for final accuracy. Furthermore, the test dataset is an unbiased set of images which we can obtain the accuracy of the network.. And this source code down below is used to train, validate, and test the video in real time on Google collaborator and our laptop. The sours code is in two separate sections.  Training the Model: YOLOv4 can be trained and built under a number of frameworks. The experimental part of our study is carried out using the Google Collaborator (Welcome to Collaborator, n.d.)It is a free cloud service based on the Jupiter notebook for artificial intelligence students and researchers. It provides a fully configured runtime for deep learning and access to a robust GPU (Graphics processing unit). For the purpose of CUDA (Compute Unified Device Architecture), which is a parallel computing platform and programming model, we used GPU, which helps to reduce the training time significantly. However, free usage has some limitations, its runtime is limited to 12 hours in a single day. Google provides drive API (Application Programming Interface) on the Jupiter notebook which we can easily use google drive as a virtual machine drive, so we can save and retrieve files from the Google drive.
