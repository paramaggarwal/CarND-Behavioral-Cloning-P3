#**Behavioral Cloning**

This project has been a great learning experience for me in actually practically working on generating my own data, architecting a good network, training a network and then testing it.

I could have focussed on any of these individual areas to get this to work very well - I have actually focussed the most on the architecture of the network itself.

I now have a successful model that laps the track with very minimal augmentation (just horizontal flips) and really tiny input images - 32x32. Also the network itself is very tiny and takes only a few seconds to train.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/data-vis.png "Data Visualization"
[image2]: ./examples/example1-all.png "Input Crop and Resize - Example 1"
[image3]: ./examples/example2-all.png "Input Crop and Resize - Example 2"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

A Convolutional Neural Network has been deployed with a small amount of image augmentation before feeding it into the network. It is a completely end-to-end system where the raw image goes in as input and the output is just the steering angle.

####2. Attempts to reduce overfitting in the model

The model has been kept really tiny to avoid over-fitting by definition.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an Adam Optimizer, so the learning rate was not tuned manually. Also, we optimise the loss calculated as the mean square deviation.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I only use the data from the center camera, plus data for recovery.

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first approach was to resize the image to something small and reuse some already known architecture like LeNet - 2 CNN layers and 2 hidden layers.

I then slowly learned that this model is more suited for a classification usecase. I replaced the max-pooling step with subsampling and the RELU activation with ELU to get more proportional activations.

The convolution steps were tuned to not throw away data in the early layers. Hence the subsampling is done only in 3rd and 4th layer. The filter size is also keep small initially so we don't miss edges of the roads etc.

Because the track will turn in one direction always, a horizontally flipped image is also fed along with the actual image. I didn't need to worry about overfitting because of the very tiny network size I restricted myself to.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle would dangerously veer close to the edge of the road, but it would recover quickly. For future enhancements, I would add more recovery data for such corner cases of the car getting too close to the edge of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

My strategy when developing networks has been to start with the most smallest one and gradually introduce complexity. I have managed to get this to work with really small number of parameters overall.

1. Layer 1 is just a 1x1 convolution filter with a depth of 3 - this acts like an auto-optimising color-space transformer - it figures out the right color-space to transform the input into.
2. Layer 2 is a 3x3 filter with depth of 3. Here I choose a small kernel size as I want to keep as much of the details as possible. The network will most likely optimise this step into an image sharpener.
3. Now we start feature extraction with two CNN layers with 5x5 kernels and depths of just 12 and 16. Here, we also sample the image by a factor of 2 on each step, reducing the size down to 1/4th in each dimension.
4. Once we are done extracting features, we flatten the image and pass it via three hidden layers of 100, 50 and 10 width.
5. The output is a single neuron.

Notice that we do not use any pooling step because of the sampling step in-built in the CNN layer. Also, the activation is an ELU (exponential) instead of RELU (rectified linear) - this is because we are not trying to do activation based classification, but instead want a proportional relationship between input and output.

####3. Creation of the Training Set & Training Process

Here is a visualization of the input data that we have. Around 8000 samples with varying steering angles. As it tends to turn in one direction more than the other, we immediately double this data by also feeding in a horizontally flipped version.

![alt text][image1]

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving along with the cropping and resizing that is done on the image.

![alt text][image2]

Here is another example of the captured input image and the crop and resize steps. Note that we are using very tiny 32x32 images as inputs to our network.

![alt text][image3]

We drive in both directions on the track to generate varying training data that will help the network generalize.

We keep training and running the network in the simulator to understand what effects do different network architectures have on the driving behavior. The architecture of this network has been explained above.

[![Video of successful lap around the track](http://img.youtube.com/vi/F7yUR7DUyTI/0.jpg)](http://www.youtube.com/watch?v=F7yUR7DUyTI "Successful Lap on the Test Track - Behavioural Cloning")

To conclude, we are now able to do successful laps without driving off the road in most cases.
