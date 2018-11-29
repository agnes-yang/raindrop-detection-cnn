################################################################################

# Example : Detect raindrops within an image by using sliding window region
# proposal algorithm and classify the ROI by using a AlexNet-30^2 CNN

# Copyright (c) 2017/18 - Tiancheng Guo / Toby Breckon, Durham University, UK

# License : https://github.com/GTC7788/raindropDetection/blob/master/LICENSE

################################################################################

# The green rectangles represents the detected raindrops.
# The red rectangles represents the ground truth raindrops in the image.

# Script takes one argument indicating the path to the set of images to process:
# e.g.
#
# python3 ./raindrop_detection_sliding_window.py -f dataset/detction/test_data/
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

from xml_parsing import *

################################################################################
# use command line parser to read command line argument for file location

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description='Rain drop classiification on a set of example images.');
parser.add_argument("-f", "--file_path", type=str, help="path to files", default='dataset/detection/image');
parser.add_argument("-gt", "--ground_truth_path", type=str, help="path to files", default='dataset/detection/ground-truth-label');
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
	network = conv_2d(network, 64, 11, strides=4, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = conv_2d(network, 128, 5, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = conv_2d(network, 256, 3, activation='relu')
	network = conv_2d(network, 256, 3, activation='relu')
	network = conv_2d(network, 128, 3, activation='relu')
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

"""
Calculates all the windows that will slide through an image.

Args:
	image: the image to apply sliding window.
	stepSize: step size (in pixel) between each window.
	windowSize: size of each window.
Return:
	All of the sliding windows for an image, each element represents
	the coordinates of top left corner of the window and its size.
"""
def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

################################################################################

"""
Sliding window algorithm will generates too many rectangles.
We can use the groupRectangles method to reduce overlapping rectangles.
Args:
	rectangleList_before: list of detected regions (rectangles).
	threshold: Minimum possible number of rectangles minus 1.
			   The threshold is used in a group of rectangles to retain it.
	eps: Relative difference between sides of the rectangles to merge them into a group.
Return:
	rectangleList_after: list of optimized detected regions.
"""
# Regularise the format of the proposed result list.
def utilize_rectangle_list(rectangleList_before, threshold, eps):
	# Using the groupRectangles() function to shrink the rectangle list
	rectangleList_after = []

	for element in rectangleList_before:
		full_rectangle_list = []
		full_rectangle_list.append(element[0])
		full_rectangle_list.append(element[1])
		full_rectangle_list.append(element[0]+30)
		full_rectangle_list.append(element[1]+30)
		rectangleList_after.append(full_rectangle_list)

	# group the proposed overlapping regions into one region,
	# decrese the recall but increase the precision.
	rectangleList_after, weight = cv2.groupRectangles(rectangleList_after, threshold, eps)

	return rectangleList_after

################################################################################

"""
Slide the window across the image, pass each window (region of interest) into the trained AlexNet.
If the region is classified as a raindrop, store the region's coordinates in a list and return
the list.

Args:
	image: the image to process
	winW: width of the sliding window
	winH: height of the sliding window
Return:
	rectangle_result: a list of region of interest that classified as raindrop by the AlexNet
"""
def cnn_find_raindrop(image, winW, winH):
	rectangle_result = []

	for (x, y, window) in sliding_window(image, stepSize=10, windowSize=(winW, winH)):
		# if the window does not meet the desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue

		# predict the region

		predict_result = model.predict([image[y:y + winH, x:x + winW]])
		final_result = np.argmax(predict_result[0]) # transfer the result to 0 or 1

		if final_result == 1:
			rectangle_result.append((x, y))
	return rectangle_result

################################################################################

# Set up the trained AlexNet-30^2

alex_net = create_basic_alexnet()
model = tflearn.DNN(alex_net)
model.load('models/alexnet_30_2_detection.tfl', weights_only = True)

# Set the height and width of the sliding window
(winW, winH) = (30, 30)

# setup display window and class names

windowName = "example image"
cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)

# show ground truth

ground_truth = True;

# process all images in directory (sorted by filename)

for filename in sorted(os.listdir(args.file_path)):

    # if it is a JPG file

    if '.jpg' in filename:
        print(os.path.join(args.file_path, filename));
        print();

        # read image

        img = cv2.imread(os.path.join(args.file_path, filename), cv2.IMREAD_COLOR)

        # convert image to RGB (from opencv BGR) and scale 0 -> 1 as per training

        rgb_image = np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype="float32")
        rgb_image /= 255

        # Get the proposed regions

        rectangle_result = cnn_find_raindrop(rgb_image, winW, winH)

        # Remove overlapping rectangles

        new_rectangle_list = utilize_rectangle_list(rectangle_result, 1, 0.1)

        # # **************** Draw Detection Rectangles - GREEN *******************

        clone = img.copy()

        for element in new_rectangle_list:
        	cv2.rectangle(clone,(element[0], element[1]),(element[2],element[3] ),(0, 255, 0),2)

        ## *************************************************************



        # ********** Draw the rectangles that contains ground truth raindrops - RED ********
        if ground_truth:
        	# Parse the xml file that contains raindrop locations of the image
        	xml_golden = parse_xml_file(os.path.join(args.ground_truth_path, filename.replace("image", "ground-truth").replace("jpg", "xml")))
        	# Read the coordinates of the raindrops
        	xml_reformat = xml_transform(xml_golden)
        	# *************** Draw the XML Result ********************
        	for element in xml_reformat:
        		cv2.rectangle(clone,(element[0], element[1]),(element[2],element[3] ),(0, 0, 255),2)
        	# ********************************************************

        # display in a window

        cv2.imshow(windowName,clone)
        key = cv2.waitKey(200) # wait 200ms
        if (key == ord('x')):
            break

# close all windows

cv2.destroyAllWindows()

################################################################################
