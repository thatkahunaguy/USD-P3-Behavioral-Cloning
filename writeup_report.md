# **Behavioral Cloning - Project #3** 

### John Glancy

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/nvidia_arch.png "Model Visualization"
[image2]: ./writeup_images/Flip_all_camera.png "Uncropped Images Augmented"
[image3]: ./writeup_images/crop_flip_all_camera.png "Cropped Images Augmented"
[image4]: ./writeup_images/recovery.jpg "Recovery Image"
[image5]: ./writeup_images/yuv.png "Blur & YUV Preprocessing"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

##### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](./model.py) containing the script to create and train the model
* [drive.py](./drive.py) for driving the car in autonomous mode
* [model.h5](./model.h5) containing a trained convolution neural network 
* [writeup_report.md](./writeup_report.md) (this file) summarizing the results

##### 2. Submission includes functional code
Using the Udacity provided simulator and my [drive.py](./drive.py) file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

##### 3. Submission code is usable and readable

The [model.py](./model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

##### 1. An appropriate model architecture has been employed

My model is identical to the Nvidia model in the [provided paper](https://arxiv.org/pdf/1604.07316.pdf) and consists of a convolutional neural network with both 5x5 and 3x3 filter sizes and depths between 24 and 64 (code lines 132-144) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 118-119). 

##### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (code lines 139-143). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 52, 107-108, 149-153). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track and a video of the successful run is included as video.mp4.

##### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (code line 148).

##### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and multiple additional runs through curves and near the small dirt sections of the track since these sections are under-represented.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

##### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to utilize the architecture outlined in [the nvidia paper](https://arxiv.org/pdf/1604.07316.pdf) since it was proven to be successful with real world driving.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set and augmented the data in a generator as described in more detail below. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to add dropout layers with 50% retention.  This successfully combatted the overfitting.

The final step was to run the simulator to see how well the car was driving around track one. Overall the vehicle drove very well with the exception of some small sections where there was dirt to the side of the road where the vehicle left the track. To improve the driving behavior in these cases, I captured additional recovery data around the dirt areas.  This helped only slightly.  I then shifted the colorspace from BGR/RGB to YUV as suggested in several forums and this resolved the issue of leaving the track near dirt sections.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

##### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network as visualized below(this is the nvidia network with droput).

![alt text][image1]

##### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded several laps on track 1 using center lane driving. In the model I augmented the data randomly in the generator to include flipped images as well as images from the right and left camera with the angle corrected to simulate center camera images.  Example of these 6 types of images are below.

![alt text][image2]

To remove artifacts unecessary for vehicle control from the image I used a Keras layer to crop the top and bottom of the images to focus on the road.  The same image set as above is shown below to demonstrate the impact of cropping.

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when it got off center and near the edges of the road.  I spent additional time capturing 10 more recoveries from the small areas with dirt at the side of the road because the model initially struggled and left the road at these sections.  The image below shows an example image of recovering from the dirt section. 

![alt text][image4]


After the collection process, I had 9153 data points. I then preprocessed this data by flipping the images and utilizing the right, left, and center cameras.  This is done randomly in the generator.  As mentioned earlier, the data is cropped to remove unecessary image data at the top and bottom of the image and is normalized.  Both of these things are done in Keras layers.  The final augmentation of the image which was done outside the Keras layers was adding a guassian blur and shifting to YUV colorspace.  This was used to correct the vehicle leaving the road in the dirt sections which still hadn't been resolved even after significant additional recovery data was added.

![alt text][image5]


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was ~5 as evidenced by very limited additional learning or validation set improvement beyond this. I used an adam optimizer so that manually training the learning rate wasn't necessary.
