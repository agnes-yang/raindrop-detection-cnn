################################################################################

# Example : Classify raindrop sampels by using a AlexNet-30^2 CNN

# Copyright (c) 2017/18 - Tiancheng Guo / Toby Breckon, Durham University, UK

# License : https://github.com/GTC7788/raindropDetection/blob/master/LICENSE

################################################################################

# Script takes one argument indicating the path to the set of images to process:
# e.g.
#
# python3 ./raindrop_classification.py -f dataset/classification/test_data/0/
#
# will process images in the specified directory printing the result for each.

################################################################################

from __future__ import division, print_function, absolute_import
import numpy as np
import cv2
import os
import argparse

################################################################################

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import build_image_dataset_from_dir
from tflearn.layers.merge_ops import merge

################################################################################
# use command line parser to read command line argument for file location

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description='Rain drop classiification on a set of example images.');
parser.add_argument("-f", "--file_path", type=str, help="path to files", default='dataset/classification/test_data/1/');
args = parser.parse_args()

################################################################################

"""
Set up the structure of AlexNet-30^2 CNN by using TFLearn.
Returns:
	network: a CNN which follows the structure of AlexNet.
"""
def create_basic_alexnet():

	# Building network as per architecture in [Guo/Breckon, 2018]

	network = input_data(shape=[None, 30, 30, 3])
	network = conv_2d(network, 96, 11, strides=4, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = conv_2d(network, 256, 5, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = conv_2d(network, 384, 3, activation='relu')
	network = conv_2d(network, 384, 3, activation='relu')
	network = conv_2d(network, 256, 3, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = fully_connected(network, 4096, activation='tanh')
	network = dropout(network, 0.5)
	network = fully_connected(network, 4096, activation='tanh')
	network = dropout(network, 0.5)
	network = fully_connected(network, 2, activation='softmax')
	network = regression(network, optimizer='momentum',
		loss='categorical_crossentropy', learning_rate=0.001)

	return network

################################################################################

# Set up the trained AlexNet-30^2

alex_net = create_basic_alexnet()
model = tflearn.DNN(alex_net)
model.load('models/alexnet_30_2_classification.tfl', weights_only = True)

# setup display window and class names

windowName = "example image"
cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)

classes = {1 : 'Raindrop', 0  : 'Non-Raindrop'}

# process all images in directory (sorted by filename)

for filename in sorted(os.listdir(args.file_path)):

    # if it is a JPG file

    if '.jpg' in filename:
        print(os.path.join(args.file_path, filename));

        # read it and display in a window

        img = cv2.imread(os.path.join(args.file_path, filename), cv2.IMREAD_COLOR)
        cv2.imshow(windowName,img)
        key = cv2.waitKey(200) # wait 200ms
        if (key == ord('x')):
            break

        # resize to 30 x 30 using LANCZOS4 interpolation (equiv PIL ANTIALIAS)

        img = cv2.resize(img, (30,30), cv2.INTER_LANCZOS4)

        # convert image to RGB (from opencv BGR) and scale 0 -> 1 as per training

        rgb_image = np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype="float32")
        rgb_image /= 255

        # pass the image into AlexNet-30^2

        predict_result = model.predict([rgb_image])

        # display result

        final_result = np.argmax(predict_result[0])

        print("--------------- result is " + str(classes[int(final_result)]) + "(class: " + str(final_result) + ")")
        print()

# close all windows

cv2.destroyAllWindows()

################################################################################
