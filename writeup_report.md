#**Behavioral Cloning** 

## Writeup Template


** Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup_report.md` summarizing the results
* `behavioral-cloning.ipynb` used to actually write all the code and visualize the data

#### 2. Submission includes functional code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The stages in which the CNN solution is built can be broken down into the usual ones: training, validation, and test -- see the diagram below. Data (camera images and steering angles) is generated using the simulator in 'training mode'. This data is preprocessed and split into training and validation sets. The training set is further augmented to counter several issues which I ran into when testing. Then the (augmented) training set is fed into the DNN for training. As a measure of progress, the DNN is tested against the validation set. When satisfied, the weights of the DNN are saved and used to run the simulator in 'autonomous mode'. There are two test tracks available where track 1 can be considered 'easier' than track 2.

![Solution Design](images/SolutionSketch.png)

I started with the well-known nvidia architecture and further evolved and adapted it to the problem at hand. The final model consists of 6 layers, three 5x5 convolutional layers, one 3x3 convolutional layer, and two fully connected layers - see the next section for further details.

For training, I first only used the training data provided by udacity making up 8'036 samples. This worked fine for track 1 but I couldn't get the model to generalized well enough to manage all of track 2 with that data. Thus, I recorded additional 15'000 samples on track 2. Half of it in standard direction, and the other half in opposite direction. Having data driving the track in both directions should allow the model to generalize better. I first used only this data for training which worked well for track 2 but the car didn't manage to drive all of track 1. Therefore, I added 11'000 samples of track 1 but only trained a few epochs with it - more on that later.

The model has about 200k parameters - about 20% less than the nvidia model which arguably was built for a much more complex task (driving in real world scenarios).
To combat overfitting, I used dropout on the fully connected layers and augmented the training data by applying random flipping, random brightness, random shifting, and random shadows.

While training the network, I monitored the mean squared error on the augmented training set and the validation set. When the mse dropped below 0.03, the model was usually capable to drive both tracks without crashing.

#### 2. Final Model Architecture

The final model architecture is

| Layer      | Description                                           | Param #   |
|:----------:|:-----------------------------------------------------:|:---------:|
| Input      | 316x76x3 RGB image                                    | 0         |
| Conv2D 5x5 | 1x1 stride, valid padding, output (312, 72, 16), relu | 1'216     |
| MaxPool2D  | 2x2 stride, output (156, 36, 16)                      | 0         |
| Conv2D 5x5 | 1x1 stride, valid padding, output (152, 32, 24), relu | 9'624     |
| MaxPool2D  | 2x2 stride, output (76, 16, 24)                       | 0         | 
| Conv2D 5x5 | 1x1 stride, valid padding, output (72, 12, 32), relu  | 19'232    |
| MaxPool2D  | 2x2 stride, output (36, 6, 32)                        | 0         | 
| Conv2D 3x3 | 1x1 stride, valid padding, output (34, 4, 48), relu   | 13'872    |
| MaxPool2D  | 2x2 stride, output (17, 2, 48)                        | 0         | 
| Flatten    | output (1632), dropout = 50 %,                        | 0         | 
| Dense      | output (100), dropout = 40 %, relu                    | 163'300   |
| Dense      | output (20), dropout = 30 %, relu                     | 2'020     |
| Dense      | output (1)                                            | 21        |

Instead of resizing the images beforehand, the convolutional+maxpool layers gradually reduce the image size picking up features on the way. Relus introduce nonlinearity into the model and to avoid overfitting of the fully connected layers dropout is used -- see the `get_model()` function for all the details.

This model has 209'000 parameters - about 20% less than the 250k parameters of the nvidia model while making use of much larger images.

Before the 320x160 images from the simulator are fed into the network, they are cropped to 316x76 pixels. Initially, the same cropping was used for training and testing in the simulator, cutting away the part of the vehicle visible in the pictures as well as most of the sky. However, I noticed that the steering got quite unstable at times. To make it easier for the model to predict a stable steering wheel, a different vertical slicing was chosen when testing to give the model a 'lookahead' advantage. After cropping, local histogram optimization is applied to compensate dark low contrast situations which are often encountered on track 2. These two steps make up all the preprocessing, everything else has been learned by the network.

The model used an adam optimizer, so the learning rate 1e-4 was not tuned manually.
To speed up learning, L2-regularization with lambda=1e-4 is used on the fully connected layers.

Here is a visualization of the architecture

![Network architecture](network.png)

#### 3. Creation of the Training Set & Training Process

In my very first attempts, I only used the training data provided by udacity making up 8'036 samples. Here are three examples

![3 examples from the udacity dataset](images/examples.PNG)

This worked fine for track 1 but I couldn't get the model to generalized well enough to manage all of track 2 with that data. Therefore, I recorded 15'316 samples on track 2. Half of it in standard direction, and the other half in opposite direction. Having data driving the track in both directions should allow the model to generalize better. Track 2 is made up of a divided lane and I drove the entire track on the right side steering with the mouse in hope to produce smoother steering data.

![3 examples from the dataset on track 2](images/examples2.PNG)

I trained the model solely with this data which worked great for driving on track 2 but there were a few issues on track 1. Those issues were mitigated by refining the weights obtained from training on track 2 data by training on additional 11'835 samples driven on track 1 - driven in opposite direction! - and only trained for 5 epochs -- more details below.

![3 examples from the dataset on track 1](images/examples3.PNG)

The unfiltered data set is biased towards steering straight with a few outliers due to extreme steering

![Histogram of unfiltered data](images/original-hist.PNG)

To have more equally distributed training data, I divided the absolute steering angle interval [0,1] into 1000 bins and chose at random at most 50 samples for each bin -- see `equalize_angles()`. Furthermore, I chose for each sample at random the image either from the center, the left, or the right camera. In case of the left and right cameras a steering correction of +10 degree and -10 degree, respectively, were added -- see `random_select_cameras`. The additional images from the left and right cameras produce samples of "driving gone wrong" which help the model to recover from errors.

![Steering correction for side cams](images/sidecams-steering-correction.PNG)

Thus ending up with a distribution which is much more balanced.

![Histogram of rectified data](images/rectified-hist.PNG)

To avoid bias towards a certain steering direction, I randomly flipped the images (together with the steering angle) -- see `apply_random_flip_single()`.

![Random flipping](images/random-flip.PNG)

To further counter steering angle bias, I also randomly shifted the images from -80px to +80px where the amount was chosen from a uniform distribution, and corrected the steering angle -- see `apply_random_shifting_single()`.

![Random shifting](images/random-shift.PNG)

While lighting on track 1 is quite uniform, track 2 has very dark and very bright spots. Therefore, I also applied random brightness augmentation to the pictures. To make the images brighter, I used gamma correction, to make them darker, I rescaled the v component in the hsv color space.

![Random brightness](images/random-brightness.PNG)

On track 2 shadows are abundant and very dark at times while on track 1 there are almost no shadows. To avoid overfitting and allow the network to generalize better to shadowish situations, I painted a black polygon with random edges and random alpha onto the pictures.

![Random shadow](images/random-shadow.PNG)

Before training, the unfiltered dataset was split 80/20 into training and validation sets, the steering angles were histogram equalized, and the set was shuffled. All augmentations mentioned above such as using left and right cameras, random flipping, random shifting, random brightness, and random shadows were only applied on the training set.

| Training Set     | Validation Set |
| ---------------- | -------------- |
| 7'432 images     | 1'858 images   |

To save memory and allow parallelization, a python generator is used, yielding from list of image paths and steering angles batches of (augmented) images & steering angles. Since all the augmentation is done on the CPU and I have a 4 core machine (with hyperthreading) the multiprocessing module was used to produce each batch of images on all cores simulatenously  -- see `generate_samples()`.

Every image of each of the sets is preprocessed by first selecting a horizontal slice from the image removing the car and most of the sky. Subsequently, skimage's local histogram optimization is applied to counter dark and low contrast situations. Then the image is fed into the network.  

![9 images from the generator (augmentation on)](images/img_generated.PNG)

The validation set helped me to determine when the model was overfitting. The ideal number of epochs was 200 when the mse on the validation set dropped below 0.3. I used an adam optimizer so that manually training the learning rate wasn't necessary.

In my testing the most difficult part of track 2 appeared to be a very sharp right turn

![turn-1](images/turn1.jpg) ![turn-2](images/turn2.jpg) ![turn-3](images/turn3.jpg)

which the model handled very well after training 200 epochs.

However, there were difficulties on track 1 which has not been seen by the model up to this point of training. In more detail, driving was okay initially staying almost centered on the lane, became slightly unstable when driving across the bridge and the car eventually didn't manage to make the sharp left turn immediately after the bridge.

To this end, I trained another 5 epochs using additional 11'837 samples driven on track 1 but in opposite direction. With this additional training, the model managed to drive all of track 1 staying in the center of the lane most of the time and never touching the lane border. Interestingly, the model afterwards struggled more with the previously mentioned sharp turn on track 2, making a lane change because of too few steering, but reverting back to the correct lane shortly afterwards.

Overall I am quite happy with the outcome not using any data driven on track 1 in the "correct" direction.

- [Video of driving track 2](t2.mp4)
- [Video of driving track 1](t1.mp4)

**Areas of potential improvements**
- Driving on track 2 is much smoother on track 1, i.e. the model still has issues to generalize smooth driving to tracks it has not seen that much
- Sharp turns on both tracks are an issue. When looking at the recorded images this may be improved by changing the FoV
- The network size could probably be further reduced by rescaling the images