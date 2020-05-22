# Lucas Kanade Tracker

[![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg)](LICENSE.md)
---


### Authors
Arpit Aggarwal
Shantam Bajpai


### Introduction to the Project
In this project, we will be using Lucas-Kanade method for object tracking in a video. Lucas Kanade states that the optical flow is essentially constant in a local neighborhood of the pixel under consideration and solves the basic optical flow equations for all the pixels in the neighborhood by the least squares criterion. The algorithm is described in more detail in the "Report.pdf" file.


### Example of Lucas-Kanade Tracker on a Car Video
![](https://j.gifs.com/5Q41kB.gif)


### Software Required
To run the .py files, use Python 3. Standard Python 3 libraries like OpenCV, Numpy, and matplotlib are used.


### Instructions for running the code
To run the code for Bolt Video, follow the following commands:

```
python Code/main_bolt.py 'images_path'
```
where, images_path is the path for bolt images. For example, running the python file on my local setup was:

```
python Code/main_bolt.py data/Bolt2/img
```


To run the code for Car Video, follow the following commands:

```
python Code/main_car4.py 'images_path'
```
where, images_path is the path for car images. For example, running the python file on my local setup was:

```
python Code/main_car4.py data/Car4/img
```


To run the code for Dragon Video, follow the following commands:

```
python Code/main_dragon.py 'images_path'
```
where, images_path is the path for dragon-baby images. For example, running the python file on my local setup was:

```
python Code/main_dragon.py data/DragonBaby/img/
```


### Credits
The following links were helpful for this project:
1. https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
2. https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2004_1/baker_simon_2004_1.pdf
3. https://www.youtube.com/watch?v=tzO245uWQxA
