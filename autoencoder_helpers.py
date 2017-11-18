# python 3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 17:01:37 2017

@author: Oscar Li
"""
import os
import tensorflow as tf

def makedirs(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def list_of_distances(X, Y):
    '''
    Given a list of vectors, X = [x_1, ..., x_n], and another list of vectors,
    Y = [y_1, ... , y_m], we return a list of vectors
            [[d(x_1, y_1), d(x_1, y_2), ... , d(x_1, y_m)],
             ...
             [d(x_n, y_1), d(x_n, y_2), ... , d(x_n, y_m)]],
    where the distance metric used is the sqared euclidean distance.
    The computation is achieved through a clever use of broadcasting.
    '''
    XX = tf.reshape(list_of_norms(X), shape=(-1, 1))
    YY = tf.reshape(list_of_norms(Y), shape=(1, -1))
    output = XX + YY - 2 * tf.matmul(X, tf.transpose(Y))

    return output

def list_of_norms(X):
    '''
    X is a list of vectors X = [x_1, ..., x_n], we return
        [d(x_1, x_1), d(x_2, x_2), ... , d(x_n, x_n)], where the distance
    function is the squared euclidean distance.
    '''
    return tf.reduce_sum(tf.pow(X, 2), axis=1)

def print_and_write(str, file):
    '''
    print str to the console and also write it to file
    '''
    print(str)
    file.write(str + '\n')

