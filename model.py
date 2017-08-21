import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import math

from sklearn.utils import shuffle
from skimage.color import rgb2gray
from skimage.exposure import equalize_adapthist, adjust_gamma
from skimage.transform import rotate, warp, ProjectiveTransform, SimilarityTransform, matrix_transform
from skimage import img_as_ubyte, img_as_float
from sklearn.model_selection import train_test_split

from multiprocessing import Pool, freeze_support

# The list recording directories
#data_paths = [ r"D:\recordings\data\data", r"D:\recordings\r1", r"D:\recordings\r2", r"D:\recordings\r3" ]
#data_paths = [ r"D:\recordings\data\data", r"D:\recordings\r2", r"D:\recordings\r3" ]
data_paths = [  r"D:\recordings\r2", r"D:\recordings\r3" ]

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
def random_select_cameras(X, y, use_side_cameras = False):
    steering_angle_corrections = [0, 10. / 180. * math.pi, -10. / 180. * math.pi]
    
    if use_side_cameras:
        ind = np.random.randint(0, 3, len(X))
        
        X_side = np.array([X[i, ind[i]] for i in range(len(X))])
        y_side = np.array([y[i] + steering_angle_corrections[ind[i]] for i in range(len(X))])
    else:
        X_side = np.array(X[:, 0], copy=True)
        y_side = np.array(y, copy=True)
        
    return X_side, y_side


def load_single(img_path):
    return np.array(img_as_float(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)), dtype=np.float32)

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
    coin = np.random.randint(0, 2) # either 0 or 1
    
    if coin == 0:
        # make it brither
        gamma = np.random.uniform(0.3, 1.0)
        
        return adjust_gamma(X, gamma)
    else:
        # make it darker
        factor = np.random.uniform(0.2, 1.0)
        
        hsv = cv2.cvtColor(X, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        v = factor * v
    
        return np.array(cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB))

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

def preprocess_sample(sample):
    X_in = sample[0]
    y = sample[1]
    augment = sample[2]

    #print(X_in.shape)
    X = load_single(X_in)[50:130, :, :]
    
    if augment:
        X = apply_random_shadow_single(X)
        X = apply_random_brightness_single(X)
        X, y = apply_random_flip_single(X, y)
        X, y = apply_random_shifting_single(X, y)
    
    X = equalize_adapthist(X)
    
    return X, y

def generate_samples(X_in, y_in, augment=False):
    batch_size = 128
    p = Pool(7)
    
    #print('samples: ', n_samples, ' bs: ', batch_size, X.shape, y.shape)
    
    while True:
        X, y = random_select_cameras(X_in, y_in, augment)
        
        n_samples = len(X)
        ind = np.random.permutation(n_samples)
        
        for batch in range(0, n_samples, batch_size):
            batch_ind = ind[batch:(batch+batch_size)]

            out = p.map(preprocess_sample, ([X[i], y[i], augment] for i in batch_ind))
            l = list(zip(*out))

            yield (np.array(l[0]), np.array(l[1]))

# ========================================================================================================
#                                     Model building
# ========================================================================================================

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

def get_model2():
    model = Sequential()
    
    reg=regularizers.l2(0.0001)

    model.add(Lambda(lambda image: image[:, 2:78, 2:318, :], input_shape=(80,320,3)))
    
    model.add(Conv2D(16, (5,5), activation='relu', strides=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (5,5), activation='relu', strides=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (5,5), activation='relu', strides=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (3,3), activation='relu', strides=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    
    model.add(Dense(500, activation='relu', kernel_regularizer=reg))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='relu',kernel_regularizer=reg))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    
    adam = Adam(lr=0.0001)
    
    model.compile(loss='mse', optimizer=adam, metrics=['mse','accuracy'])
    
    return model

def get_simple_model():
    model = Sequential()
    
    model.add(Lambda(lambda image: ktf.image.resize_images(image, (32, 128)), input_shape=(80,320,3)))
    
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(500, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    
    adam = Adam(lr=0.0001)
    
    model.compile(loss='mse', optimizer=adam, metrics=['mse','accuracy'])
    
    return model

def get_nvidia_model():
    model = Sequential()
    
    model.add(Lambda(lambda image: ktf.image.resize_images(image, (66, 200)),input_shape=(80,320,3)))
    
    model.add(BatchNormalization(epsilon=0.001, axis=1))

    model.add(Conv2D(24,(5,5),padding='valid', activation='relu', strides=(2,2)))
    model.add(Conv2D(36,(5,5),padding='valid', activation='relu', strides=(2,2)))
    model.add(Conv2D(48,(5,5),padding='valid', activation='relu', strides=(2,2)))
    model.add(Conv2D(64,(3,3),padding='valid', activation='relu', strides=(1,1)))
    model.add(Conv2D(64,(3,3),padding='valid', activation='relu', strides=(1,1)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    
    adam = Adam(lr=0.0001)
    
    model.compile(loss='mse', optimizer=adam, metrics=['mse','accuracy'])
    
    return model

# ========================================================================================================
#                                     Run the code!
# ========================================================================================================

if __name__ == '__main__':
    from keras.models import Sequential, load_model
    from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, Activation, Lambda, Cropping2D
    from keras.optimizers import Adam
    from keras.layers import BatchNormalization,Input
    from keras import regularizers
    from keras.backend import tf as ktf
    from keras.callbacks import ModelCheckpoint

    freeze_support()

    # Build the keras model and print a summary
    #model = get_simple_model()
    #model = get_model()
    #model = load_model('model-my.h5', custom_objects={"ktf": ktf})
    #model = get_nvidia_model()
    #model = load_model('nv-weights\model-09.h5', custom_objects={"ktf": ktf})
    #model = load_model('model-my2.h5', custom_objects={"ktf": ktf})
    model = get_model2()
    model.summary()

    # Parse the csv files
    X_data, y_data = parse_recordings(data_paths)
    X_eq, y_eq = equalize_angles(X_data, y_data)

    # Train / Validation split
    X_train, X_val, y_train, y_val = train_test_split(X_eq, y_eq, test_size=.2)

    # Train the model

    model.fit_generator(generate_samples(X_train, y_train, True), 
                    steps_per_epoch=len(X_train)/128,
                    epochs=200,
                    validation_data=generate_samples(X_val, y_val, False),
                    validation_steps = len(X_val)/128,
                    callbacks=[ModelCheckpoint(r'D:\Projects\CarND-Behavioral-Cloning-P3\noob-model\modelx-{epoch:02d}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)])

    # Save the trained model
    model.save('noob-modelx.h5')