# Image Classification Using Deep Learning in Go

There is a spectre haunting data scientists who program their deep learning models in R and Python-- the spectre of deep learning in Go.

github repo link: https://github.com/jssuzuki1/deep_learning_with_go

# Directory Overview

- "data" contains all of the image data.
- "results" contains the exports of main.go in the code folder..
- "main.exe" is the executable of main.go in the code folder.

In the 'code' folder:
- "main.go" is the main program that loads in the image data and feeds it to the deep learning models.
- "main_test.go" is a test program of the main program, which is a unit test for importing data in the form expected of MNIST data. The test should return as a success with the terminal command "go test -v"

# Data

This exercises uses the MNIST data set, which is comprised of images of hand-drawn numbers in grey scale along with their true classification called "labels." Each image pixel is represented with a numerical value between 0 and 255 that represents the shade of light for a number. Each image is an array of these values.
This exercises uses the MNIST data set, which is comprised of images of hand-drawn numbers in grey scale along with their true classification called "labels." Each image pixel is represented with a numerical value between 0 and 255 that represents the shade of light for a number. Each image is an array of these values.

# Deep Learning Method

I decided upon gorgonia as my package of choice because *Hands-on Deep Learning with Go. A Practical Guide to Building and Implementing Neural Network Models* uses it. I attempted to implement it, but ran into some issues regarding the shape of the input data when I attempted a forward pass.

# Methodology

The general idea is to use deep learing to predict the numerical representations of each image to predict the digit it was intended to represent. Therefore, the labels for each representation are 'target values.'

All of the images are loaded into Go. To train the deep learning model, there are 60,000 images and labels in the training set and 10,000 images in the test set. The training set contains both the numerical representations and their target values. This training set is fed to the deep learning model to determine its model parameters.

With the model trained, it is applied to the target values to predict the classification of each image.

# Results

As expected, many of the images were misclassified by the deep learning model. Some of these incorrectly predicted images could be attributed to the fact that some of the test hand-drawn images were outlier. 

All of the image records were (attempted to be) classified typical or outliers via an isolation forest. 

The Isolation Forest is an unsupervised learing algorithm that detects anomalous data patterns. The high-level idea is that, by partitioning the data repeatedly on a random subset of data on each iteration, the algorithm hones in on the "isolated" points. Generally, a smaller number of partitions is required to find outliers because they are rare and different from other instances.

Its output is a list of scores assigned to each record, where the negative scores are outliers.

From the deep learning model, there were [x number] of images that were misclassified. Of these, [x number] were outliers identified by the isolation forest. These [x numbers] were numbers I was planning to fill in if my model had worked.

# Conclusion and Applications to OCR

This study's scope is within digit recognition. However, a similar method could be performed for OCR purposes.

With a large body of PDFs and their true text, a deep learing model could be trained to read other PDFs. These PDFs could be of varying lengths, font sizes, and such to ensure that whatever deep learning model that is used can identify those aspects.


# Other repositories programmed in Go by the author

Forgive many of the generic names, as these repositories were completed Assignments in grad school. 

https://github.com/jssuzuki1/go_assignment_benchmark
- A benchmark test for the basic linear regression

https://github.com/jssuzuki1/assignment_2f
- A website via Hugo, a web development framework

https://github.com/jssuzuki1/command_line_application
- A command line application that returns different metrics (min, mean, or max on a specified column) on housing prices in Chicago depending on user input.

https://github.com/jssuzuki1/assignment_5a
- A webscraper of Wikipedia articles.

https://github.com/jssuzuki1/Regression_Concurrency
- A benchmark test of 100 iterations of linear regressions, comparing the standard iterative loop with the 100 iterations run with concurrency.

https://github.com/jssuzuki1/assignment_7a 
- Compares running an isolation in python versus Go

https://github.com/jssuzuki1/wails_vale_svelte_demo
- A desktop application that attempts to correct uses of the word "data" with the singular use of the word "datum."

https://github.com/jssuzuki1/wales_sqlite_chatbot
- A desktop application that serves as a basic chatbot.