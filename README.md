# On the Impact of Varying Region Proposal Strategies for Raindrop Detection and Classification using Convolutional Neural Networks

Tested using Python 3.4.6, [TensorFlow 1.9.0](https://www.tensorflow.org/install/), and [OpenCV 3.3.1](http://www.opencv.org)

(requires opencv extra modules - ximgproc module for superpixel segmentation)

## Region Proposal Strategies:

![Demonstration of Raindrop Detection](https://github.com/tobybreckon/raindropDetection/blob/master/images/example-all.png)

Raindrop detection vis differing region proposal strategies (sliding window, superpixel and selective search)

## Abstract
_"The presence of raindrop induced image distortion has a significant negative impact on the performance of a wide
range of all-weather visual sensing applications including within the increasingly important contexts of visual
surveillance and vehicle autonomy. A key part of this problem is robust raindrop detection such that the potential
for performance degradation in effected image regions can be identified. Here we address the problem of raindrop
detection in colour video imagery by considering three varying region proposal approaches with secondary classification
via a number of novel convolutional neural network architecture variants. This is verified over an extensive dataset
with in-frame raindrop annotation to achieve maximal 0.95 detection accuracy with minimal false positives compared to
prior work (1). Our approach is evaluated under a range of environmental conditions typical of all-weather automotive
visual sensing applications."_

(1) using Alexnet-30^2 CNN model

[[Guo and Breckon, In Proc. International Conference on Image Processing IEEE, 2018](https://breckon.org/toby/publications/papers/guo18raindrop.pdf)]

---

## Reference implementation

This raindrop detection approach was based on various region proposal strategies with optimal classification via the down-scaled AlexNet model (Alexnet-30^2)

This repository contains ```raindrop_classification.py```, ```raindrop_detection_sliding_window.py``` and ```raindrop_detection_super_pixel.py``` files
corresponding to raindrop classification based on Alexnet-30^2, raindrop detection based on exhaustive search via sliding window and superpixel from the paper, as these approaches
demonstrate the best accuracy as shown in the paper.

To use these scripts the pre-trained network models must be downloaded using the shell script ```download-models.sh``` which will create an additional ```models``` directory containing the network weight data.
Furthermore, the example dataset must be downloaded using the shell script ```download-data.sh``` which will create an additional ```dataset``` directory containing the image data for testing.

The custom dataset used for training and evaluation can be found on [Durham Collections](https://collections.durham.ac.uk/collections/r2c534fn94m) (together with the trained network models):

-

- 

---

## Instructions to test pre-trained models:

```
$ git clone https://github.com/tobybreckon/raindrop-detection-cnn.git
$ cd raindrop-detection-cnn
$ sh ./download-models.sh
$ sh ./download-dataset.sh

$ # to test positive classification test examples
$ python raindrop_classification.py -f dataset/classification/test_data/1

$ # to to test sliding window detection and classification
$ python raindrop_detection_sliding_window.py -f dataset/detection/images/

$ # to follow
$ python raindrop_detection_super_pixel.py 3

```

---

## Example Video

[![Examples]()]()

< to follow> Video Example for Raindrop Detection with Sliding Window - click image above to play.

[![Examples]()]()

< to follow> Video Example for Raindrop Detection with Super Pixel - click image above to play.

---

## Reference

[On the impact of varying region proposal strategies for raindrop detection and classification using convolutional neural networks](http://breckon.eu/toby/publications/papers/guo18raindrop.pdf)
(Guo, Akcay, Adey and Breckon), In Proc. International Conference on Image Processing IEEE, 2018.
```
@InProceedings{guo18raindrop,
  author =     {Guo, T. and Akcay, S. and Adey, P. and Breckon, T.P.},
  title =      {On the impact of varying region proposal strategies for raindrop detection and classification using convolutional neural networks},
  booktitle =  {Proc. International Conference on Image Processing},
  pages =      {1-5},
  year =       {2018},
  month =      {September},
  publisher =  {IEEE},
  keywords =   {rain detection, raindrop distortion, all-weather computer vision, automotive vision, CNN},
}

```

---
