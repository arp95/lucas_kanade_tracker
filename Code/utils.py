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
import math
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

# initialise step of em algorithm
def initialise_step(n, d, k, data):
    """
    Inputs:
    n - number of datapoints
    d - dimension of the gaussian
    k - number of the gaussians
    
    Outputs:
    weights_gaussian - weight of the gaussians, size (k)
    mean_gaussian - mean of the gaussians, size (k x d)
    covariance_matrix_gaussian - covariance of the gaussians, size (k x d x d)
    probability_values - probability of the datapoint being in the k-gaussians, size (n x k)
    """
    
    # initialise weights
    weights_gaussian = np.zeros(k)
    for index in range(0, k):
        weights_gaussian[index] = (1.0 / k)
    
    # initialise mean
    mean_gaussian = np.zeros((k, d))
    for dimension in range(0, k):
        mean_gaussian[dimension] = np.array([data[np.random.choice(data.shape[0])]], np.float64)
    
    # initialise covariance
    covariance_matrix_gaussian = np.zeros((k, d, d))
    for dimension in range(0, k):
        covariance_matrix_gaussian[dimension] = np.matrix(np.multiply([np.random.randint(1,255) * np.eye(data.shape[1])], np.random.rand(data.shape[1], data.shape[1])))
    
    # randomly initialise probability
    probability_values = np.zeros((n, k))
    for index in range(0, n):
        probability_values[index][np.random.randint(0, k)] = 1
        
    # return the arrays
    return (weights_gaussian, mean_gaussian, covariance_matrix_gaussian, probability_values)


# gaussian estimation for expectation step
def gaussian_estimation(data_point, mean, covariance, dimension):
    """
    Inputs:
    data_point - data point of the gaussian, size (1 x d)
    mean - mean of the gaussian, size (1 x d)
    covariance - covariance of the gaussian, size (1 x d x d)
    dimension - dimension of the gaussian
    
    Outputs:
    value of the gaussian
    """
    
    determinant_covariance = np.linalg.det(covariance)
    determinant_covariance_root = np.sqrt(determinant_covariance)
    covariance_inverse = np.linalg.inv(covariance)
    gaussian_pi_coeff = 1.0 / np.power((2 * np.pi), (dimension / 2))
    data_mean_diff = (data_point - mean)
    data_mean_diff_transpose = data_mean_diff.T     
    return (gaussian_pi_coeff) * (determinant_covariance_root) * np.exp(-0.5 * np.matmul(np.matmul(data_mean_diff, covariance_inverse), data_mean_diff_transpose))


# gaussian estimation for n-points
def gaussian_estimation_array(data_point, mean, covariance, dimension):
    """
    Inputs:
    data_point - data point of the gaussian, size (n x d)
    mean - mean of the gaussian, size (1 x d)
    covariance - covariance of the gaussian, size (1 x d x d)
    dimension - dimension of the gaussian
    
    Outputs:
    value of the gaussian, size (n x d)
    """
    
    determinant_covariance = np.linalg.det(covariance)
    determinant_covariance_root = np.sqrt(determinant_covariance)
    covariance_inverse = np.linalg.inv(covariance)
    gaussian_pi_coeff = 1.0 / np.power((2 * np.pi), (dimension / 2))
    data_mean_diff = (data_point - mean)
    data_mean_diff_transpose = data_mean_diff.T 
    val = (gaussian_pi_coeff) * (determinant_covariance_root) * np.exp(-0.5 * np.sum(np.multiply(data_mean_diff * covariance_inverse, data_mean_diff), axis=1))
    return np.reshape(val, (data_point.shape[0], data_point.shape[1]))


# gaussian estimation for 3-dimensional case
def gaussian_estimation_3d(data_point, mean, cov):
    """
    Inputs:
    data_point - data point of the gaussian, size (n x d)
    mean - mean of the gaussian, size (1 x d)
    cov - covariance of the gaussian, size (1 x d x d)
    
    Outputs:
    value of the gaussian, size (n x d)
    """

    det_cov = np.linalg.det(cov)
    cov_inv = np.zeros_like(cov)
    mean = np.array(mean)
    cov = np.array(cov)
    for i in range(data_point.shape[1]):
        cov_inv[i, i] = 1 / cov[i, i]
    diff = np.matrix(data_point - mean)
    return (2.0 * np.pi) ** (-len(data_point[1]) / 2.0) * (1.0 / (np.linalg.det(cov) ** 0.5)) * np.exp(-0.5 * np.sum(np.multiply(diff * cov_inv, diff), axis=1))


def compute_log_likelihood(input_data, mixture_coeff, mean_gaussian, covariance_gaussian, K):
    
    """
    Inputs
    
    mixture_coeff: Mixture coefficient is the probability of the kth gaussian of size (k)
    mean_gaussian: the mean is the mean of the kth gaussian of size (kxd)
    covariance_gaussian: Covariance of the Kth gaussian of size (kxdxd)
    data - training data, size (n x d)
    K: Number of Gaussians
    
    """
    # We need to compute the log likelihood and do a test for convergence
    n = input_data.shape[0]

    log_likelihood =  np.log(np.sum(mixture_coeff[k]*mvn.pdf(input_data,mean_gaussian[k],covariance_gaussian[k]) for k in range(K)))
        
    return np.sum(log_likelihood)


# e-step of the algorithm
# reference: https://towardsdatascience.com/an-intuitive-guide-to-expected-maximation-em-algorithm-e1eb93648ce9
def expectation_step(n, d, k, data, weights_gaussian, mean_gaussian, covariance_matrix_gaussian, probability_values):
    """
    Inputs:
    n - the number of data-points
    d - dimension of gaussian
    k - number of gaussians
    data - data to be trained on of size (n x d)
    weights_gaussian - weight of gaussians of size (k)
    mean_gaussian - mean of gaussians of size (k x d)
    covariance_matrix_gaussian - covariance of gaussians of size (k x d x d)
    probability_values - probability of the datapoint being in a gaussian of size (n x k)
    
    Outputs:
    probabilities - probability array of size (n x k)
    """
    
    # create empty array of list of probabilities
    for dimension in range(0, k):
        probability_values[: ,dimension:dimension+1] = gaussian_estimation_3d(data, mean_gaussian[dimension], covariance_matrix_gaussian[dimension]) * weights_gaussian[dimension]
            
    prob_sum = np.sum(probability_values, axis=1)
    log_likelihood = np.sum(np.log(prob_sum))
    probability_values = np.divide(probability_values, np.tile(prob_sum, (k,1)).transpose())
    return (np.array(probability_values), log_likelihood)


# m-step of the algorithm
# reference: https://towardsdatascience.com/an-intuitive-guide-to-expected-maximation-em-algorithm-e1eb93648ce9
def maximization_step(n, d, k, data, weights_gaussian, mean_gaussian, covariance_matrix_gaussian, probability_values):
    """
    Inputs:
    n - number of data-points
    d - dimension of gaussian
    k - number of gaussians
    data - training data, size (n x d)
    weights_gaussian - weight of the gaussians, size (k)
    mean_gaussian - mean of the gaussians, size (k x d)
    covariance_matrix_gaussian - covariance of the gaussians, size (k x d x d)
    probability_values - probability of the datapoint being in a gaussian, size (n x k)
    
    Outputs:
    weights_gaussian - weight of the gaussians, size (k)
    new_mean - mean of the gaussians, size (k x d)
    new_cov - covariance of the gaussians, size (k x d x d)
    """

    prob_sum = np.sum(probability_values, axis=0)
    new_mean = np.zeros_like(mean_gaussian)
    new_cov = np.zeros_like(covariance_matrix_gaussian)
    for dimension in range(0, k):
        temp_sum = math.fsum(probability_values[:, dimension])
        new_mean[dimension] = 1. / prob_sum[dimension] * np.sum(probability_values[:, dimension] * data.T, axis = 1).T           
        diff = data - new_mean[dimension]
        new_cov[dimension] = np.array(1. / prob_sum[dimension] * np.dot(np.multiply(diff.T, probability_values[:, dimension]), diff)) 
        weights_gaussian[dimension] = 1. / (data.shape[0]) * prob_sum[dimension]
    return (weights_gaussian, new_mean, new_cov)


# run e-m algorithm
def run_expectation_maximization_algorithm(n, d, k, iterations, data):
    """
    Inputs:
    n - number of data-points
    d - dimension of gaussian
    k - number of gaussians
    iterations - number of iterations 
    data - training data, size (n x d)
    
    Outputs:
    weights_gaussian - weight of the gaussians, size (k)
    mean_gaussian - mean of the gaussians, size (k x d)
    covariance_matrix_gaussian - covariance of the gaussians, size (k x d x d)
    """
    
    # initialise step
    (weights_gaussian, mean_gaussian, covariance_matrix_gaussian, probability_values) = initialise_step(n, d, k, data)
    log_likelihoods = []    

    # run for fixed iterations
    for i in range(0, iterations):
    
        # m-step
        (weights_gaussian, mean_gaussian, covariance_matrix_gaussian) = maximization_step(n, d, k, data, weights_gaussian, mean_gaussian, covariance_matrix_gaussian, probability_values)
    
        # e-step
        probability_values, log_likelihood = expectation_step(n, d, k, data, weights_gaussian, mean_gaussian, covariance_matrix_gaussian, probability_values)
        log_likelihoods.append(log_likelihood)    

        # Loss Function( Test for convergence )
        log_likelihood = compute_log_likelihood(data, weights_gaussian, mean_gaussian, covariance_matrix_gaussian,k)
        
        if i%10 == 0:
            print("For iteration: " + str(i) + "\nweights: " + str(weights_gaussian) + "\nmean_gaussian: " + str(mean_gaussian) + "\ncovariance: " + str(covariance_matrix_gaussian) + "\nloss: " + str(log_likelihood))
            print()
        if(len(log_likelihoods) > 2 and np.abs(log_likelihood - log_likelihoods[-2]) < 0.0001):
            break
  
    # return answer
    return (weights_gaussian, mean_gaussian, covariance_matrix_gaussian)


# plot histogram
def plot_hist(image):
    # loop over the image channels
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    features = []
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        features.extend(hist)
        plt.plot(hist, color = color)
        plt.xlim([0, 256])


# data for training
def get_training_data(file_path, channel1, channel2, channel3):
    data = []
    files = glob.glob(file_path + "/*")
    for file in files:
        image = cv2.imread(file)
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                val = []
                if(channel1):
                    val.append(image[row, col, 0])
                if(channel2):
                    val.append(image[row, col, 1])
                if(channel3):
                    val.append(image[row, col, 2])
                data.append(val)
    return np.array(data)


# reference: https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
        
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)
