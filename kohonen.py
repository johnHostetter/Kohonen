#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 21:01:53 2021

@author: john
"""

import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def a(t):
    return pow(1.025, -t)

def self_organizing_learning(X, num_of_terms_per_antecedent):
    """
    An adapted version of Kohonen's feature-maps algorithm that obtains approximate
    centers and widths for linguistic terms before subsequent learning phases.

    Parameters
    ----------
    X : Numpy 2-D array
        A matrix containing entries with shape (T, n1) where T is the number of observations and n1 is the number of inputs.
    num_of_terms_per_antecedent : list
        A list containing entries describing the number of linguistic terms per antecedent
        (e.g. if the first entry of the list (at index 0) is 3, then the first antecedent has 3 linguistic terms).

    Returns
    -------
    None.

    """
    # display debug console information
    VERBOSE = False
    # the number of inputs into the system
    n1 = len(num_of_terms_per_antecedent)
    # create a 2-D matrix to store the c_{i, j} values where i is input, j is term 
    # (not all entries will be used if some inputs have less terms than others)
    # shape = (n1, max(num_of_terms_per_antecedent)) # this is the "correct" shape, but has to be flipped to generate random centers correctly
    shape = (max(num_of_terms_per_antecedent), n1)
    centers = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), shape).T # the transpose flips the matrix to match the paper
    widths = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), shape).T # the transpose flips the matrix to match the paper
    if VERBOSE:
        print('original shape (only to create the centers matrix): %s' % str(shape))
        print('centers: %s' % centers)
    shape = centers.shape
    if VERBOSE:
        print('new shape (the correct shape of the centers matrix): %s' % str(shape))
    T = len(X)
    
    for t, x in enumerate(X):
        if VERBOSE:
            print('t = %s, x = %s' % (t, x))
        for i, T_i in enumerate(num_of_terms_per_antecedent):
            j = np.argmin(x[i] - centers[i])
            if VERBOSE:
                print('i = %s, T_i = %s' % (i, T_i))
                print(x[i])
                print(centers[i])
                print('diff = %s' % (x[i] - centers[i]))
                print('j = %s' % j)
                print(centers)
            centers[i, j] = centers[i, j] + a(t) * (x[i] - centers[i, j])
            
    for i, T_i in enumerate(num_of_terms_per_antecedent):
        for j in range(T_i):
            diff_vector = centers[i] - centers[i, j]
            closest_nonzero_center = np.min(diff_vector[np.nonzero(diff_vector)])
            widths[i, j] = np.abs(centers[i, j] - closest_nonzero_center) / 2
    
    print(centers)
    print(widths)
    for idx, centers_with_widths in enumerate(zip(centers, widths)):
        cs = centers_with_widths[0]
        ws = centers_with_widths[1]
        for j, c in enumerate(cs):
            mu = c
            sigma = ws[j]
            print(mu, sigma)
            if sigma != 0:
                x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
                plt.plot(x, gaussian(x, mu, sigma))
        plt.show()
    
    return centers, widths