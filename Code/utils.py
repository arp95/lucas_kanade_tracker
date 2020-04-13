"""
 *  MIT License
 *
 *  Copyright (c) 2019 Arpit Aggarwal Shantam Bajpai
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a
 *  copy of this software and associated documentation files (the "Software"),
 *  to deal in the Software without restriction, including without
 *  limitation the rights to use, copy, modify, merge, publish, distribute,
 *  sublicense, and/or sell copies of the Software, and to permit persons to
 *  whom the Software is furnished to do so, subject to the following
 *  conditions:
 *
 *  The above copyright notice and this permission notice shall be included
 *  in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 *  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *  DEALINGS IN THE SOFTWARE.
"""


# header files
import glob
import cv2
import numpy as np


# get the grayscale image after converting it to lab colorspace
def get_grayscale_image(image):
    """
    Inputs:
    
    image: the input color image
    
    Outputs:
    
    gray: grayscale image
    """
    
    # convert to lab space, apply clahe to b channel and then get grayscale image
    #clahe = cv2.createCLAHE(clipLimit = 1.0, tileGridSize = (1, 1))
    #lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    #l, a, b = cv2.split(lab)
    #lab = cv2.merge((l, a, clahe.apply(b)))
    #lab = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # return grayscale image
    return gray


# update the grayscale image, that is, normalize the pixel values with template image
def update_grayscale_image(template_image, image):
    """
    Inputs:
    
    template_image: the template grayscale image whose ROI is given to us
    image: the current grayscale frame 
    
    Outputs:
    
    image: the normalized current grayscale frame
    """
    
    # get the mean of the template image and current frame and normalize
    template_mean = np.mean(template_image)
    mean = np.mean(image)
    image = (image * (template_mean / mean)).astype(float)
    
    # return the normalized current grayscale frame
    return image


# computes error between images
def compute_error_images(image_array1, image_array2):
    """
    Inputs:
    
    image_array1: the template image array
    image_array2: the current frame array
    
    Outputs:
    
    error: the error between each pixel of the template image and the current image
    """
    
    return (image_array1 - image_array2)


# computes the coordinates of the ROI of interest
def get_coordinates_array(x_range, y_range):
    """
    Inputs:
    
    x_range: array of two elements consisting of min_x and max_x of the ROI region
    y_range: array of two elements consisting of min_y and max_y of the ROI region
    
    Outputs:
    
    coordinates_array: array of all the coordinates in the ROI region of the image
    """
    
    # create coordinates array and push each possible coordinate in the ROI region
    coordinates_array = np.zeros((3, int((x_range[1] - x_range[0] + 1) * (y_range[1] - y_range[0] + 1))))
    count = 0
    for y in range(int(y_range[0]), int(y_range[1]) + 1):
        for x in range(int(x_range[0]), int(x_range[1]) + 1):
            coordinates_array[0, count] = x
            coordinates_array[1, count] = y
            coordinates_array[2, count] = 1
            count = count + 1
            
    # return the array
    return coordinates_array


# computes the new coordinates of the image wrt template image
def get_new_image_coordinates(template_image_coordinates_array, p, x_range, y_range):
    """
    Inputs:
    
    template_image_coordinates_array: the coordinates of the ROI region of template image
    p: the matrix to calculate the warped image
    x_range: the range of x values for the image
    y_range: the range of y values for the image
    
    Outputs:
    
    new_gray_image_coordinates_array: the current frame ROI coordinates
    new_rectangle_coordinates: the current frame rectangle coordinates (four: top-left, bottom-left, bottom-right, top-right)
    """
    
    # define the rectangle coordinates of the ROI of template image
    template_rectangle_coordinates = np.array([[x_range[0], x_range[0], x_range[1], x_range[1]], [y_range[0], y_range[1], y_range[1], y_range[0]], [1, 1, 1, 1]])

    # get the affine matrix to get the warped image
    affine_matrix = np.zeros((2, 3))
    count = 0
    for i in range(0, 3):
        for j in range(0, 2):
            affine_matrix[j, i] = p[count, 0]
            count = count + 1
    affine_matrix[0, 0] = affine_matrix[0, 0] + 1
    affine_matrix[1, 1] = affine_matrix[1, 1] + 1
    
    # get new rectange coordinates
    new_rectangle_coordinates = np.dot(affine_matrix, template_rectangle_coordinates)
    
    # get new coordinates
    new_gray_image_coordinates_array = np.dot(affine_matrix, template_image_coordinates_array)
    new_gray_image_coordinates_array = new_gray_image_coordinates_array.astype(int)
    
    # return the two arrays
    return (new_gray_image_coordinates_array, new_rectangle_coordinates)
    
    
# computes the pixel array
def get_pixel_array(image, image_coordinates_array):
    """
    Inputs:
    
    image: the input image
    image_coordinates_array: the ROI region in the image
    
    Outputs:
    
    pixel_array: the array consisting of pixels of the ROI region of the image
    """
    
    # get the pixel values of the ROI of the image
    pixel_array = np.zeros((1, int(image_coordinates_array.shape[1])))
    pixel_array[0, :] = image[image_coordinates_array[1, :], image_coordinates_array[0, :]]
    
    # return the pixel array
    return pixel_array
    
    
# compute the error in the p matrix
def get_delta_p(error, steep_descent):
    """
    Inputs:
    
    error: the error between images
    steep_descent: the steep descent value
    
    Outputs:
    delta_p: the change in the p matrix
    """
    
    # compute the sd_param and hessian matrix
    sd_param_matrix = np.dot(steep_descent.T, error.T)
    hessian_matrix = np.dot(steep_descent.T, steep_descent)
    hessian_matrix_inv = np.linalg.pinv(hessian_matrix)
    
    # use the above two matrices to get the error in p matrix and return
    delta_p = np.dot(hessian_matrix_inv, sd_param_matrix)
    return delta_p    
    
    
# compute the steep descent using two images and the coordinates of the ROI of two images
def get_steep_descent(sobelx, sobely, template_image_coordinates_array, new_gray_image_coordinates_array):
    """
    Inputs:
    
    sobelx: the derivative along x-direction
    sobely: the derivative along y-direction
    template_gray_image_coordinates_array: the ROI coordinates of the template image
    new_gray_image_coordinates_array: the ROI coordinates of the current frame
    
    Outputs:
    
    image: 6 images formed using above information
    """
    
    # get the pixel array for sobelx
    sobelx_pixel_array = get_pixel_array(sobelx, new_gray_image_coordinates_array)
    
    # get the pixel array for sobely
    sobely_pixel_array = get_pixel_array(sobely, new_gray_image_coordinates_array)
    
    # get four images
    image1 = sobelx_pixel_array * template_image_coordinates_array[0, :]
    image2 = sobely_pixel_array * template_image_coordinates_array[0, :]
    image3 = sobelx_pixel_array * template_image_coordinates_array[1, :]
    image4 = sobely_pixel_array * template_image_coordinates_array[1, :]
    
    # return the six images
    return np.vstack((image1, image2, image3, image4, sobelx_pixel_array, sobely_pixel_array)).T
    
    
# lucas kanade algorithm
def lucas_kanade_algorithm(template_gray_image, current_gray_image, x_range, y_range, thresh, p):
    """
    Inputs:
    
    template_gray_image: the grayscale template image
    current_gray_image: the grayscale current frame
    x_range: the array consisting of min_x and max_x values
    y_range: the array consisting of min_y and max_y values
    thresh: the threshold after which we need to break the loop
    p: the matrix used for calculating the warped image
    
    Outputs:
    
    new_rectangle_coordinates: the new ROI for the current frame
    """
    
    # get the coordinates of the ROI for template image
    template_image_coordinates_array = get_coordinates_array(x_range, y_range)
    
    # define p matrix, used for calculating the warped image for template image
    p_template = np.array([[0, 0, 0, 0, 0, 0]]).T
    
    # get the coordinates of the ROI in the new frame
    (new_template_image_coordinates_array, new_rectangle_coordinates) = get_new_image_coordinates(template_image_coordinates_array, p_template, x_range, y_range)
    
    # get the pixel array of the template image
    template_pixel_array = get_pixel_array(template_gray_image, new_template_image_coordinates_array)
    
    # compute derivatives around x and y directions for the current frame
    sobelx = cv2.Sobel(current_gray_image, cv2.CV_64F, 1, 0, ksize = 3)
    sobely = cv2.Sobel(current_gray_image, cv2.CV_64F, 0, 1, ksize = 3)
    
    # run algorithm
    count = 0
    while(True):
        
        # get the coordinates of the ROI in the new frame
        (new_gray_image_coordinates_array, new_rectangle_coordinates) = get_new_image_coordinates(template_image_coordinates_array, p, x_range, y_range)
        
        # if new coordinates not in range, then break
        if(count > 10 or new_gray_image_coordinates_array[0, 0] < 0 or new_gray_image_coordinates_array[1, 0] < 0 or new_gray_image_coordinates_array[0, new_gray_image_coordinates_array.shape[1] - 1] > current_gray_image.shape[1] or new_gray_image_coordinates_array[1, new_gray_image_coordinates_array.shape[1] - 1] > current_gray_image.shape[0]):
            break
            
        # get the pixel array of the gray image
        new_pixel_array = get_pixel_array(current_gray_image, new_gray_image_coordinates_array)
        
        # compute the error
        error = compute_error_images(template_pixel_array, new_pixel_array)
        
        # compute steep descent
        steep_descent = get_steep_descent(sobelx, sobely, template_image_coordinates_array, new_gray_image_coordinates_array)
        
        # get the delta_p
        delta_p = get_delta_p(error, steep_descent)
        
        # get p norm and update p matrix
        p_norm = np.linalg.norm(delta_p)
        p = np.reshape(p, (6, 1))
        p = p + delta_p
        count = count + 1
        
        # if p_norm within thresh break
        if(p_norm < thresh):
            break
            
    # return the ROI needed for this frame
    return (new_rectangle_coordinates, p)
