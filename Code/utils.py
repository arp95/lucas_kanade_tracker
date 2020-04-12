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
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1, 1))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lab = cv2.merge((l, a, clahe.apply(b)))
    lab = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
    return gray

# update the grayscale image, that is, normalize the pixel values with template image
def update_grayscale_image(template_image, image):
    template_mean = np.mean(template_image)
    mean = np.mean(image)
    image = (image * (template_mean / mean)).astype(float)
    return image

# computes error between images
def compute_error_images(image_array1, image_array2):
    return (image_array1 - image_array2)

# computes the coordinates of the ROI of interest
def get_coordinates_array(x_range, y_range):
    coordinates_array = np.zeros(3, (x_range[1] - x_range[0] + 1) * (y_range[1] - y_range[0] + 1))
    count = 0
    for y in range(y_range[0], y_range[1] + 1):
        for x in range(x_range[0], x_range[1] + 1):
            coordinates_array[0, count] = x
            coordinates_array[1, count] = y
            coordinates_array[2, count] = 1
            count = count + 1
    return coordinates_array

# computes the new coordinates of the image wrt template image
def get_new_image_coordinates(template_gray_image_coordinates_array, p, x_range, y_range):
    
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
    new_gray_image_coordinates_array = np.dot(affine_matrix, template_gray_image_coordinates_array)
    new_gray_image_coordinates_array = new_gray_image_coordinates_array.astype(int)
    
    # return the two arrays
    return (new_gray_image_coordinates_array, new_rectangle_coordinates)
    
# computes the pixel array
def get_pixel_array(image, image_coordinates_array):
    
    # get the pixel values of the ROI of the image
    pixel_array = np.zeros(1, image_coordinates_array.shape[1])
    pixel_array[0, :] = image[image_coordinates_array[1, :], image_coordinates_array[0, :]]
    return pixel_array
    
# compute the error in the p matrix
def get_delta_p(error, steep_descent):
    # compute the sd_param and hessian matrix
    sd_param_matrix = np.dot(steep_descent.T, error.T)
    hessian_matrix = np.dot(steep_descent.T, steep_descent)
    hessian_matrix_inv = np.linalg.pinv(hessian_matrix)
    
    # use the above two matrices to get the error in p matrix and return
    delta_p = np.dot(hessian_matrix_inv, sd_param_matrix)
    return delta_p    
    
# compute the steep descent using two images and the coordinates of the ROI of two images
def get_steep_descent(sobelx, sobely, template_gray_image_coordinates_array, new_gray_image_coordinates_array):
    
    # get the pixel array for sobelx
    sobelx_pixel_array = get_pixel_array(sobelx, new_gray_image_coordinates_array)
    
    # get the pixel array for sobely
    sobely_pixel_array = get_pixel_array(sobely, new_gray_image_coordinates_array)
    
    # get four images
    image1 = sobelx_pixel_array * template_gray_image_coordinates_array[0, :]
    image2 = sobely_pixel_array * template_gray_image_coordinates_array[0, :]
    image3 = sobelx_pixel_array * template_gray_image_coordinates_array[1, :]
    image4 = sobely_pixel_array * template_gray_image_coordinates_array[1, :]
    
    # return the six images
    return np.vstack((image1, image2, image3, image4, sobelx_pixel_array, sobely_pixel_array)).T
    
# lucas kanade algorithm
def lucas_kanade_algorithm(template_gray_image, current_gray_image, sobelx, sobely, x_range, y_range, thresh):
    
    # get the coordinates of the ROI for template image
    template_gray_image_coordinates_array = get_coordinates_array(x_range, y_range)
    
    # define p matrix, used for calculating the warped image
    p = np.array([[0, 0, 0, 0, 0, 0]]).T
    
    # get the coordinates of the ROI in the new frame
    (new_template_image_coordinates_array, new_rectangle_coordinates) = get_new_image_coordinates(template_gray_image_coordinates_array, p, x_range, y_range)
    
    # get the pixel array of the template image
    template_pixel_array = get_pixel_array(template_gray_image, new_template_image_coordinates_array)
    
    # run algorithm
    while(True):
        
        # get the coordinates of the ROI in the new frame
        (new_gray_image_coordinates_array, new_rectangle_coordinates) = get_new_image_coordinates(template_gray_image_coordinates_array, p, x_range, y_range)
        
        # if new coordinates not in range, then break
        if(new_gray_image_coordinates_array[0, 0] < 0 or new_gray_image_coordinates_array[1, 0] < 0 or new_gray_image_coordinates_array[0, new_gray_image_coordinates_array.shape[1] - 1] > current_gray_image.shape[1] or new_gray_image_coordinates_array[1, new_gray_image_coordinates_array.shape[1] - 1] > current_gray_image.shape[0]):
            break
            
        # get the pixel array of the gray image
        new_pixel_array = get_pixel_array(current_gray_image, new_gray_image_coordinates_array)
        
        # compute the error
        error = compute_error_images(template_pixel_array, new_pixel_array)
        
        # compute steep descent
        steep_descent = get_steep_descent(sobelx, sobely, template_gray_image_coordinates_array, new_gray_image_coordinates_array)
        
        # get the delta_p
        delta_p = get_delta_p(error, steep_descent)
        
        # get p norm and update p
        p_norm = np.linalg.norm(delta_p)
        p = np.reshape(p, (6, 1))
        p = p + delta_p
        
        # if p_norm within thresh break
        if(p_norm < thresh):
            break
            
    # return the ROI needed for this frame
    return new_rectangle_coordinates
