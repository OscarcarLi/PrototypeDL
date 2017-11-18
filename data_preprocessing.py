# python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 21:22:58 2017
"""
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def batch_elastic_transform(images, sigma, alpha, height, width, random_state=None):
    '''
    this code is borrowed from chsasank on GitHubGist
    Elastic deformation of images as described in [Simard 2003].
    
    images: a two-dimensional numpy array; we can think of it as a list of flattened images
    sigma: the real-valued variance of the gaussian kernel
    alpha: a real-value that is multiplied onto the displacement fields
    
    returns: an elastically distorted image of the same shape
    '''
    assert len(images.shape) == 2
    # the two lines below ensure we do not alter the array images
    e_images = np.empty_like(images)
    e_images[:] = images
    
    e_images = e_images.reshape(-1, height, width)
    
    if random_state is None:
        random_state = np.random.RandomState(None)
    x, y = np.mgrid[0:height, 0:width]
    
    for i in range(e_images.shape[0]):
        
        dx = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma, mode='constant') * alpha
        dy = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma, mode='constant') * alpha
        indices = x + dx, y + dy
        e_images[i] = map_coordinates(e_images[i], indices, order=1)

    return e_images.reshape(-1, 784)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tensorflow.examples.tutorials.mnist import input_data
    '''
    the following code demonstrates how gaussian_filter works by ploting
    the displacement field before and after applying the gaussian_filter
    '''
    random_state = np.random.RandomState(None)
    dx1 = random_state.rand(28, 28) * 2 - 1
    dy1 = random_state.rand(28, 28) * 2 - 1
    dx2 = gaussian_filter(dx1, 4, mode='constant')
    dy2 = gaussian_filter(dy1, 4, mode='constant')
    x, y = np.mgrid[0:28, 0:28]
    plt.quiver(x, y, dx1, dy1)
    plt.show()
    plt.quiver(x, y, dx2, dy2)
    plt.show()
    
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    img = mnist.train.images[0]
    plt.imshow(img.reshape(28, -1), cmap='gray')
    plt.show()
    dimg = elastic_transform(img.reshape(28, -1), sigma=4, alpha=20)
    plt.imshow(dimg, cmap='gray')
    plt.show()
    plt.close()