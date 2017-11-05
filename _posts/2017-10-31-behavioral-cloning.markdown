---
layout: post
title:  "Behavioral Cloning"
date:   2017-11-01 20:25:27 +0100
---

# **Behavioral Cloning** 

[architect]: /assets/model_architecture.png "Model Visualization"
[explo1]: /assets/data_exploration1.png "Data Exploration: Before Augmentation"
[explo2]: /assets/data_exploration2.png "Data Exploration: After Augmentation"
[augment]: /assets/data_augment1.png "Data Augmentation"
[train_valid]: /assets/train_valid.png "Train/Validate loss"
[code]: https://github.com/robroooh/CarND-Behavioral-Cloning-P3/blob/master/behavioral_cloning.ipynb

<iframe width="560" height="315" src="https://www.youtube.com/embed/Wtxhrw1ssmU" frameborder="0" allowfullscreen></iframe>

#### 1. Model architecture

My model is based on the [nvidia's paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
here. Using 4 convolutional layers with ELU activation to introduce non-linearity ([code] line 178).
The data was normalized in the model using a Keras lambda layer ([code] line 172).


#### 2. Reducing overfitting in the model

The model contains dropout layers in order to reduce overfitting ([code] lines 194). 

The dataset was splitted to 80/20 train/validate dataset, so that the model won't overfitted ([code] line 65)

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was automatically tuned ([code] line 210).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used udacity's provided data because it was challenging
to make use of existing data and augmented them to solve the task. 
The data is from normally driving, and it provides left/right camera images which
able us to teach the car to learn from mistakes.

The details on data augmentation can be seen in the next section.
 
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The strategy used for deriving an architecture was to make the network that could
predict the angle solely from an image.

I did not start building the architecture from scratch. In fact, I used a cnn the exact same network as
in the [nvidia's paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

The main reasons I thought this model was appropriate are that:
1. It combined many convolutional layers with appropriate sizes for feature engineering.
2. It was proved by visualization of the activation of feature maps in the paper that the architecture is 
able to learn useful features on its own.

To test how cool my model was, I splitted the dataset (image, steering_angle) into training/validation set.

The loss of my model was too high, that i decided to work more on data augmentation & tuning left/right camera offset.

Moreover, I add [dropout layers to prevent complex co-adaptation](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) & overfitting 
between each of last 3 Fully Connected layers

The final step was to run the simulator to see how well the car was driving around track one. At first, the car always went off the track.
I notice this bahavior as it does not know how to get back to the center of lane.

To improve the driving behavior in these cases, I fine-tuned the left/right camera offset.
I found that 0.2 was an acceptable number which I could let the car drive for hours without going off track despite some jittery.

#### 2. Final Model Architecture

Here is a (long) visualization of the architecture

![alternate][architect]

My model can be describe in details by the following list (on model.py lines xx-xx) :

 1. Lambda layer - for normalization
 1. Cropping Layer - crop 50px from the top, 20px from the bottom
 1. 24@5x5 stride-2 convolutional layer following by ELU Activation
 1. 36@5x5 stride-2 convolutional layer following by ELU Activation
 1. 48@5x5 stride-2 convolutional layer following by ELU Activation
 1. 64@3x3 stride-1 convolutional layer following by ELU Activation
 1. Flatten layer
 1. 1000 Fully Connected layer following by ELU Activation
 1. Dropout with p=0.5
 1. 100 Fully Connected layer following by ELU Activation
 1. Dropout with p=0.5
 1. 50 Fully Connected layer following by ELU Activation
 1. Dropout with p=0.5
 1. 1 Fully Connected layer

#### 3. Creation of the Training Set & Training Process


To create a training set, I am more interested in using
udacity's provided dataset than collecting my own dataset 
because of the following reasons
 - the dataset that udacity provides are small, left-biased, and 0-angles biased.
 Thus, it can't be used directly.
 Trying to make it work involves much effort on understanding the data, task, 
 and architecture which I found challenging.
 - the udacity's task state that to recover the car from off-center to center can be
 done by collecting data which a car drives from the corner to the center, but I like
 how nvidia works on the offset adjustment for left/right camera images which is perfect
 for me to make the most out of existing data.

To get myself into udacity's data, I first explore the data

![alt text][explo1]
*The histrogram of the udacity's dataset: the X-axis is the steering_angle 
and the Y-axis is the number that element on X-axis occurs (plot with bin=50)*

>I noticed some biased on the 0 angle, so I know that I need to balance the data.
Meanwhile I perceived the numbers of data for other angles are too small.
Data Augmentation is definitely needed here

![alt text][augment]

My Augmentation set here is:
 1. Offset images from left/center/right camera
    - additional off-center shifts of images are then used 
    for network to learn how to recover from mistakes or it will 
    slowly drive off the center, the magnitude of 
    shift value seem to be 0.2. I tried(0.08, 0.1, 0.2) the last one was the best,
    but I'm sure it can be better
 1. Resize the images to (128,128)
    - the numbers are from eyeballing. I tried 64,64 but I think there must 
  be loss of information, so I try larger image sizes that my eyes & 
  training speed feel comfortable with
 1. Flip
     - both the image along the horizontal-axis and angle
 1. Random brightness **or** shadow on both flipped and non-flipped images
     - brightness adjustment 
         - by adjusting gain in the range of 0.3-0.6
     - shadows shade
         - random vertical line with the width of 40 pixels
 1. Cropping
     - the top for 50 pixels
     - the bottom for 20 pixels
 1. Randomly select only 10% of 0.0 angles dataset available

Here is the result of after implementing data augmentation pipeline, there are much more 
data available to train from ~8k to ~50k

![alt text][explo2]

However, the data is too big to fit in the memory, so I use Keras's generator() to
do data real-time augmentation while training the model.

I finally randomly shuffled the data set and put 20% of the data into a validation set.
The validation set contains only image from center camera and no augmentation was done. 

I used this training data for training the model. 
The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was 60 as evidenced by it seems that another 10 epochs
might make the network overfitted.

![alt text][train_valid]

I used an adam optimizer, so no learning rate was adjusted.