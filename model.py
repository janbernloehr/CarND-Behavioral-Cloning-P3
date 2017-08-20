import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

from sklearn.utils import shuffle
from skimage.color import rgb2gray
from skimage.exposure import equalize_adapthist, adjust_gamma
from skimage.transform import rotate, warp, ProjectiveTransform, SimilarityTransform, matrix_transform
from skimage import img_as_ubyte, img_as_float
from sklearn.model_selection import train_test_split

# The list recording directories
data_paths = [ r"D:\recordings\data\data", r"D:\recordings\r1", r'D:\recordings\r4' ] 

# ========================================================================================================
#                                     raw data loading section
# ========================================================================================================

# Parses the recorded csv files and returns a tuple of image paths (center, left, right) and corresponding 
# steering angles 
def parse_recordings(paths):
    lines = []
    
    for data_path in paths:
        with open(os.path.join(data_path, r'driving_log.csv')) as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None) # skip headers
            
            for line in reader:
                for i in range(3):
                    # Rebuild the file name because the absolute paths in driving_log are
                    # not portable
                    line[i] = os.path.join(data_path, 'IMG', os.path.basename(line[i]))
                
                lines.append(line)
    
    # N x 3 Matrix containing the image paths (center, left, right) 
    images = []
    # N x 1 Matrix containing the steering while angles
    measurements = []

    for line in lines[1:]:
        images.append([line[0], line[1], line[2]])
        
        measurement = float(line[3])
        measurements.append(measurement)
        
    images = np.array(images)
    measurements = np.array(measurements)
    
    return images, measurements

# Equalizes the given X, y samples using histogram randomization
def equalize_angles(X, y, n_bins = 1200, max_number = 50):
    X_out = []
    y_out = []
    
    start = 0
    
    # Partition the interval [0, 1.2] into n_bins bins and
    # choose at random at most max_number of representatives
    for end in np.linspace(0, 1.2, n_bins):
        ind = [i for i in range(len(X)) if abs(y[i]) >= start and abs(y[i]) < end]
        
        if len(ind) > max_number:
            ind = np.random.choice(ind, max_number)
        
        X_out.append(X[ind])
        y_out.append(y[ind])
        
        start = end
        
    return np.concatenate(X_out, axis=0), np.concatenate(y_out, axis=0)

# Takes X = Nx3 and y = Nx1 as input. If use_side_cameras is False,
# only the center images are returned (X = Nx1, y = Nx1).
# If use_side_cameras is true, then also the images from the left and right
# camera are added to the list of returned images with a corresponding steering
# wheel correction
def select_cameras(X, y, use_side_cameras = False):
    steering_angle_correction = 10. / 180. * math.pi # 10 deg

    X_side = np.array(X[:, 0], copy=True)
    if use_side_cameras:
        X_side = np.append(X_side, X[:, 1], axis = 0)
        X_side = np.append(X_side, X[:, 2], axis = 0)

    y_side = np.array(y, copy=True)
    if use_side_cameras:
        y_side = np.append(y_side, y + steering_angle_correction, axis = 0)
        y_side = np.append(y_side, y - steering_angle_correction, axis = 0)
    
    return X_side, y_side

# Load all images from the given list. Expects images to be encoded as BGR. Converts to RGB float32 image 
def load_all_imgs(img_paths):
    images = []
    
    for img_path in img_paths:
        images.append(img_as_float(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)))
        
    return np.array(images, dtype=np.float32)

# ========================================================================================================
#                                     data augmentation section
# ========================================================================================================

# Apply random flipping to the given image (and apply steering wheel correction)
def apply_random_flip_single(X, y):
    flip = np.random.randint(-1, 2)
    
    if flip < 0:
        X = X[:, ::-1, :]
        y = -y
            
    return X, y

# Apply random brightness change to given image
def apply_random_brightness_single(X):
    gamma = np.random.uniform(0.5, 1.5)
    
    return adjust_gamma(X, gamma)

# Apply random shifting to the given image and apply steering wheel correction
# max_delta = Maximum number of pixels to shift
# ratio*delta = steering correction
def apply_random_shifting_single(X, y, max_delta = 80, ratio = 0.001):
    delta = np.random.uniform(-max_delta, max_delta)
        
    offsetx = delta
    offsety = 0
    
    tf_shift = SimilarityTransform(translation=[offsetx, offsety])
    
    X_copy = warp(X, tf_shift)
    y += -ratio*delta
    
    return X_copy, y

# Draw a random shadow on a given image
def apply_random_shadow_single(X):
    img_width, img_height = X.shape[0:2]
    
    alpha = np.random.uniform(0.1, 0.85)
    x = np.random.uniform(0, img_width, 2)
    width = np.random.uniform(80, 240, 2)
    
    points = np.array([[x[0],0], [x[1],img_height], [min(img_width,x[1] + width[1]),img_height], [min(x[0]+width[0],img_width),0]], np.int32)
    points = points.reshape((-1,1,2))
    shadow_color = (0,0,0)
    
    X_copy = cv2.cvtColor(X, cv2.COLOR_RGB2RGBA)
    X_overlay = np.array(X_copy, copy=True)
    X_overlay = cv2.fillPoly(X_overlay, [points], shadow_color)
    X_temp = np.empty(X_copy.shape, dtype=np.float32)
    
    cv2.addWeighted(X_copy, alpha, X_overlay, 1.-alpha, 0, X_temp)
    
    return cv2.cvtColor(X_temp, cv2.COLOR_RGBA2RGB)

# ========================================================================================================
#                                     Training data generation
# ========================================================================================================

def generate_samples(X, y, augment=False):
    # apply random steering angle histogram equalization
    X, y = equalize_angles(X, y)
    X, y = select_cameras(X, y, augment)
    
    n_samples = len(X)
    batch_size = 128
    ind = np.random.permutation(n_samples)
    count = 0
    
    #print('samples: ', n_samples, ' bs: ', batch_size, X.shape, y.shape)
    
    while True:
        for batch in range(0, n_samples, batch_size):
            batch_ind = ind[batch:(batch+batch_size)]
            
            # load images from disk
            X_b = load_all(X[batch_ind])[:, 50:130, :, :]
            y_b = np.array(y[batch_ind], copy=True)
            
            if augment:
                for i in range(len(batch_ind)):
                    X_b[i], y_b[i] = apply_random_flip_single(X_b[i], y_b[i])
                    X_b[i] = apply_random_brightness_single(X_b[i])
                    X_b[i], y_b[i] = apply_random_shifting_single(X_b[i], y_b[i])
                    X_b[i] = apply_random_shadow_single(X_b[i])

            # Apply local histogram equalization to fight low contrast  
            for i in range(len(batch_ind)):
                X_b[i] = equalize_adapthist(X_b[i])
                
            count += len(batch_ind)
                    
            yield (X_b, y_b)

# ========================================================================================================
#                                     Model building
# ========================================================================================================

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, Activation, Lambda, Cropping2D
from keras.optimizers import Adam
from keras.layers import BatchNormalization,Input
from keras import regularizers
from keras.backend import tf as ktf

def get_model():
    model = Sequential()
    
    reg=regularizers.l2(0.0001)
    
    model.add(Conv2D(32, (5,5), activation='relu', strides=(1,1), input_shape=(80,320,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (5,5), activation='relu', strides=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (5,5), activation='relu', strides=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    
    model.add(Dense(1000, activation='relu', kernel_regularizer=reg))
    model.add(Dropout(0.5))
    model.add(Dense(25, activation='relu',kernel_regularizer=reg))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    
    adam = Adam(lr=0.0001)
    
    model.compile(loss='mse', optimizer=adam, metrics=['mse','accuracy'])
    
    return model

# ========================================================================================================
#                                     Run the code!
# ========================================================================================================

# Build the keras model and print a summary
model = get_model()
model.summary()

# Parse the csv files
X_data, y_data = parse_recordings(data_paths)

# Train / Validation split
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=.2)

# Train the model
model.fit_generator(generate_samples(X_train, y_train, True), 
                    steps_per_epoch=int(len(X_train))/128,
                    epochs=10,
                    validation_data=generate_samples(X_val, y_val, False),
                    validation_steps = int(len(X_val))/128)

# Save the trained model
model.save('model.h5')