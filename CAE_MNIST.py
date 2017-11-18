# python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:23:31 2017

@author: Oscar Li
"""
from __future__ import division, print_function, absolute_import
import os
import time
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from autoencoder_helpers import makedirs, list_of_distances, print_and_write, list_of_norms
from data_preprocessing import batch_elastic_transform

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

GPUID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

# the directory to save the model
model_folder = os.path.join(os.getcwd(), "saved_model", "mnist_model", "mnist_cae_1")
makedirs(model_folder)
img_folder = os.path.join(model_folder, "img")
makedirs(img_folder)
model_filename = "mnist_cae"
# the maximum number of model snapshots we allow tensorflow to save to disk
# when set to None there is no limit
n_saves = None
# console_log is the handle to a text file that records the console output
console_log = open(os.path.join(model_folder, "console_log.txt"), "w+")

# training parameters
learning_rate = 0.002
training_epochs = 1500
batch_size = 250          # the size of a minibatch
test_display_step = 100   # how many epochs we do evaluate on the test set once
save_step = 50            # how frequently do we save the model to disk

# elastic deformation parameters
sigma = 4
alpha = 20

# lambda's are the ratios between the four error terms
lambda_class = 20
lambda_ae = 1
lambda_1 = 1              # 1 and 2 here corresponds to the notation we used in the paper
lambda_2 = 1


input_height = 28         # MNIST data input shape
input_width = input_height
n_input_channel = 1       # the number of color channels; for MNIST is 1.
input_size = input_height * input_width * n_input_channel   # the number of pixels in one input image
n_classes = 10

# Network Parameters
n_prototypes = 15         # the number of prototypes
n_layers = 4

# height and width of each layers' filters
f_1 = 3
f_2 = 3
f_3 = 3
f_4 = 3

# stride size in each direction for each of the layers
s_1 = 2
s_2 = 2
s_3 = 2
s_4 = 2

# number of feature maps in each layer
n_map_1 = 32
n_map_2 = 32
n_map_3 = 32
n_map_4 = 10

# the shapes of each layer's filter
filter_shape_1 = [f_1, f_1, n_input_channel, n_map_1]
filter_shape_2 = [f_2, f_2, n_map_1, n_map_2]
filter_shape_3 = [f_3, f_3, n_map_2, n_map_3]
filter_shape_4 = [f_4, f_4, n_map_3, n_map_4]

stride_1 = [1, s_1, s_1, 1]
stride_2 = [1, s_2, s_2, 1]
stride_3 = [1, s_3, s_3, 1]
stride_4 = [1, s_4, s_4, 1]

# tf Graph input
# X is the 2-dimensional matrix whose every row is an image example.
# Y is the 2-dimensional matrix whose every row is the one-hot encoding label.
X = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name='X')
X_img = tf.reshape(X, shape=[-1, input_height, input_width, n_input_channel], name='X_img')
Y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='Y')

# We create a tf placeholder for every lambda so that they can be tweaked during training
lambda_class_t = tf.placeholder(dtype=tf.float32, shape=(), name="lambda_class_t")
lambda_ae_t = tf.placeholder(dtype=tf.float32, shape=(), name="lambda_ae_t")
lambda_2_t = tf.placeholder(dtype=tf.float32, shape=(), name="lambda_2_t")
lambda_1_t = tf.placeholder(dtype=tf.float32, shape=(), name="lambda_1_t")

weights = {
    'enc_f1': tf.Variable(tf.random_normal(filter_shape_1,
                                           stddev=0.01,
                                           dtype=tf.float32),
                          name='encoder_f1'),
    'enc_f2': tf.Variable(tf.random_normal(filter_shape_2,
                                           stddev=0.01,
                                           dtype=tf.float32),
                          name='encoder_f2'),
    'enc_f3': tf.Variable(tf.random_normal(filter_shape_3,
                                           stddev=0.01,
                                           dtype=tf.float32),
                          name='encoder_f3'),
    'enc_f4': tf.Variable(tf.random_normal(filter_shape_4,
                                           stddev=0.01,
                                           dtype=tf.float32),
                          name='encoder_f4'),
    'dec_f4': tf.Variable(tf.random_normal(filter_shape_4,
                                           stddev=0.01,
                                           dtype=tf.float32),
                          name='decoder_f4'),
    'dec_f3': tf.Variable(tf.random_normal(filter_shape_3,
                                           stddev=0.01,
                                           dtype=tf.float32),
                          name='decoder_f3'),
    'dec_f2': tf.Variable(tf.random_normal(filter_shape_2,
                                           stddev=0.01,
                                           dtype=tf.float32),
                          name='decoder_f2'),
    'dec_f1': tf.Variable(tf.random_normal(filter_shape_1,
                                           stddev=0.01,
                                           dtype=tf.float32),
                          name='decoder_f1')
}

biases = {
    'enc_b1': tf.Variable(tf.zeros([n_map_1], dtype=tf.float32),
                          name='encoder_b1'),
    'enc_b2': tf.Variable(tf.zeros([n_map_2], dtype=tf.float32),
                          name='encoder_b2'),
    'enc_b3': tf.Variable(tf.zeros([n_map_3], dtype=tf.float32),
                          name='encoder_b3'),
    'enc_b4': tf.Variable(tf.zeros([n_map_4], dtype=tf.float32),
                          name='encoder_b4'),
    'dec_b4': tf.Variable(tf.zeros([n_map_3], dtype=tf.float32),
                          name='decoder_b4'),
    'dec_b3': tf.Variable(tf.zeros([n_map_2], dtype=tf.float32),
                          name='decoder_b3'),
    'dec_b2': tf.Variable(tf.zeros([n_map_1], dtype=tf.float32),
                          name='decoder_b2'),
    'dec_b1': tf.Variable(tf.zeros([n_input_channel], dtype=tf.float32),
                          name='decoder_b1')
}

last_layer = {
    'w': tf.Variable(tf.random_uniform(shape=[n_prototypes, n_classes],
                                       dtype=tf.float32),
                     name='last_layer_w')
}

# padding can be either "SAME" or "VALID"
def conv_layer(input, filter, bias, strides, padding="VALID", nonlinearity=tf.nn.relu):
    conv = tf.nn.conv2d(input, filter, strides=strides, padding=padding)
    act = nonlinearity(conv + bias)
    return act

# tensorflow's conv2d_transpose needs to know the shape of the output
def deconv_layer(input, filter, bias, output_shape, strides, padding="VALID", nonlinearity=tf.nn.relu):
    deconv = tf.nn.conv2d_transpose(input, filter, output_shape, strides, padding=padding)
    act = nonlinearity(deconv + bias)
    return act

def fc_layer(input, weight, bias, nonlinearity=tf.nn.relu):
    return nonlinearity(tf.matmul(input, weight) + bias)

# construct the model
# eln means the output of the nth layer of the encoder
el1 = conv_layer(X_img, weights['enc_f1'], biases['enc_b1'], stride_1, "SAME")
el2 = conv_layer(el1, weights['enc_f2'], biases['enc_b2'], stride_2, "SAME")
el3 = conv_layer(el2, weights['enc_f3'], biases['enc_b3'], stride_3, "SAME")
el4 = conv_layer(el3, weights['enc_f4'], biases['enc_b4'], stride_4, "SAME")

# we compute the output shape of each layer because the deconv_layer function requires it
l1_shape = el1.get_shape().as_list()
l2_shape = el2.get_shape().as_list()
l3_shape = el3.get_shape().as_list()
l4_shape = el4.get_shape().as_list()

flatten_size = l4_shape[1] * l4_shape[2] * l4_shape[3]
n_features = flatten_size
# feature vectors is the flattened output of the encoder
feature_vectors = tf.reshape(el4, shape=[-1, flatten_size], name='feature_vectors')

# the list prototype feature vectors
prototype_feature_vectors = tf.Variable(tf.random_uniform(shape=[n_prototypes, n_features],
                                                          dtype=tf.float32),
                                        name='prototype_feature_vectors')

'''deconv_batch_size is the number of feature vectors in the batch going into
the deconvolutional network. This is required by the signature of
conv2d_transpose. But instead of feeding in the value, the size is infered during
sess.run by looking at how many rows the feature_vectors matrix has
'''
deconv_batch_size = tf.identity(tf.shape(feature_vectors)[0], name="deconv_batch_size")

# this is necessary for prototype images evaluation
reshape_feature_vectors = tf.reshape(feature_vectors, shape=[-1, l4_shape[1], l4_shape[2], l4_shape[3]])

# dln means the output of the nth layer of the decoder
dl4 = deconv_layer(reshape_feature_vectors, weights['dec_f4'], biases['dec_b4'],
                   output_shape=[deconv_batch_size, l3_shape[1], l3_shape[2], l3_shape[3]],
                   strides=stride_4, padding="SAME")
dl3 = deconv_layer(dl4, weights['dec_f3'], biases['dec_b3'],
                   output_shape=[deconv_batch_size, l2_shape[1], l2_shape[2], l2_shape[3]],
                   strides=stride_3, padding="SAME")
dl2 = deconv_layer(dl3, weights['dec_f2'], biases['dec_b2'],
                   output_shape=[deconv_batch_size, l1_shape[1], l1_shape[2], l1_shape[3]],
                   strides=stride_2, padding="SAME")
dl1 = deconv_layer(dl2, weights['dec_f1'], biases['dec_b1'],
                   output_shape=[deconv_batch_size, input_height, input_width, n_input_channel],
                   strides=stride_1, padding="SAME", nonlinearity=tf.nn.sigmoid)
'''
X_decoded is the decoding of the encoded feature vectors in X;
we reshape it to match the shape of the training input
X_true is the correct output for the autoencoder
'''
X_decoded = tf.reshape(dl1, shape=[-1, input_size], name='X_decoded')
X_true = tf.identity(X, name='X_true')

'''
prototype_distances is the list of distances from each x_i to every prototype
in the latent space
feature_vector_distances is the list of distances from each prototype to every x_i
in the latent space
'''
prototype_distances = list_of_distances(feature_vectors,
                                        prototype_feature_vectors)
prototype_distances = tf.identity(prototype_distances, name='prototype_distances')
feature_vector_distances = list_of_distances(prototype_feature_vectors,
                                             feature_vectors)
feature_vector_distances = tf.identity(feature_vector_distances, name='feature_vector_distances')

# the logits are the weighted sum of distances from prototype_distances
logits = tf.matmul(prototype_distances, last_layer['w'], name='logits')
probability_distribution = tf.nn.softmax(logits=logits,
                                         name='probability_distribution')

'''
the error function consists of 4 terms, the autoencoder loss,
the classification loss, and the two requirements that every feature vector in
X look like at least one of the prototype feature vectors and every prototype
feature vector look like at least one of the feature vectors in X.
'''
ae_error = tf.reduce_mean(list_of_norms(X_decoded - X_true), name='ae_error')
class_error = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=logits)
class_error = tf.identity(class_error, name='class_error')
error_1 = tf.reduce_mean(tf.reduce_min(feature_vector_distances, axis = 1), name='error_1')
error_2 = tf.reduce_mean(tf.reduce_min(prototype_distances, axis = 1), name='error_2')

# total_error is the our minimization objective
total_error = lambda_class_t * class_error +\
              lambda_ae_t * ae_error + \
              lambda_1_t * error_1 + \
              lambda_2_t * error_2
total_error = tf.identity(total_error, name='total_error')

# accuracy is not the classification error term; it is the percentage accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1),
                              tf.argmax(Y, 1),
                              name='correct_prediction')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32),
                          name='accuracy')

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_error)
#add the optimizer to collection so that we can retrieve the optimizer and resume training
tf.add_to_collection("optimizer", optimizer)
    
# Create the variable init operation and a saver object to store the model
init = tf.global_variables_initializer()

hyperparameters = {
	"learning_rate": learning_rate,
    "training_epochs": training_epochs,
    "batch_size": batch_size,
    "test_display_step": test_display_step,
    "save_step": save_step,

    "lambda_class": lambda_class,
    "lambda_ae": lambda_ae,
    "lambda_1": lambda_1,
    "lambda_2": lambda_2,

    "input_height": input_height,
    "input_width": input_width,
    "n_input_channel": n_input_channel,
    "input_size": input_size,
    "n_classes": n_classes,

    "n_prototypes": n_prototypes,
    "n_layers": n_layers,

    "f_1":	f_1,
    "f_2":	f_2,
    "f_3": 	f_3,
    "f_4": 	f_4,

    "s_1" :s_1,
    "s_2": s_2,
    "s_3": s_3,
    "s_4": s_4,

    "n_map_1": n_map_1,
    "n_map_2": n_map_2,
    "n_map_3": n_map_3,
    "n_map_4": n_map_4,

    "n_features": n_features,
}
# save the hyperparameters above in the model snapshot
for (name, value) in hyperparameters.items():
    tf.add_to_collection('hyperparameters', tf.constant(name=name, value=value))

saver = tf.train.Saver(max_to_keep=n_saves)

config = tf.ConfigProto()
# the amount of GPU memory our process occupies
config.gpu_options.per_process_gpu_memory_fraction = 0.3
with tf.Session(config=config) as sess:
    sess.run(init)
    # we compute the number of batches because both training and evaluation
    # happens batch by batch; we do not throw the entire test set onto the GPU
    n_train_batch = mnist.train.num_examples // batch_size
    n_valid_batch = mnist.validation.num_examples // batch_size
    n_test_batch = mnist.test.num_examples // batch_size
    # Training cycle
    for epoch in range(training_epochs):
        print_and_write("#"*80, console_log)
        print_and_write("Epoch: %04d" % (epoch), console_log)
        start_time = time.time()
        train_ce, train_ae, train_e1, train_e2, train_te, train_ac = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        # Loop over all batches
        for i in range(n_train_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            elastic_batch_x = batch_elastic_transform(batch_x, sigma=sigma, alpha=alpha, height=input_height, width=input_width)
            _, ce, ae, e1, e2, te, ac = sess.run(
                                    (optimizer,
                                    class_error,
                                    ae_error,
                                    error_1,
                                    error_2,
                                    total_error,
                                    accuracy),
                                    feed_dict={X: elastic_batch_x,
                                               Y: batch_y,
                                               lambda_class_t: lambda_class,
                                               lambda_ae_t: lambda_ae,
                                               lambda_1_t: lambda_1,
                                               lambda_2_t: lambda_2})
            train_ce += (ce/n_train_batch)
            train_ae += (ae/n_train_batch)
            train_e1 += (e1/n_train_batch)
            train_e2 += (e2/n_train_batch)
            train_te += (te/n_train_batch)
            train_ac += (ac/n_train_batch)
        end_time = time.time()
        print_and_write('training takes {0:.2f} seconds.'.format((end_time - start_time)), console_log)
        # after every epoch, check the error terms on the entire training set
        print_and_write("training set errors:", console_log)
        print_and_write("\tclassification error: {:.6f}".format(train_ce), console_log)
        print_and_write("\tautoencoder error: {:.6f}".format(train_ae), console_log)
        print_and_write("\terror_1: {:.6f}".format(train_e1), console_log)
        print_and_write("\terror_2: {:.6f}".format(train_e2), console_log)
        print_and_write("\ttotal error: {:.6f}".format(train_te), console_log)
        print_and_write("\taccuracy: {:.4f}".format(train_ac), console_log)

        # validation set error terms evaluation
        valid_ce, valid_ae, valid_e1, valid_e2, valid_te, valid_ac = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        # Loop over all batches
        for i in range(n_valid_batch):
            batch_x, batch_y = mnist.validation.next_batch(batch_size)
            ce, ae, e1, e2, te, ac = sess.run(
                                    (class_error,
                                    ae_error,
                                    error_1,
                                    error_2,
                                    total_error,
                                    accuracy),
                                    feed_dict={X: batch_x,
                                               Y: batch_y,
                                               lambda_class_t: lambda_class,
                                               lambda_ae_t: lambda_ae,
                                               lambda_2_t: lambda_2,
                                               lambda_1_t: lambda_1})
            valid_ce += ce/n_valid_batch
            valid_ae += ae/n_valid_batch
            valid_e1 += e1/n_valid_batch
            valid_e2 += e2/n_valid_batch
            valid_te += te/n_valid_batch
            valid_ac += ac/n_valid_batch

        # after every epoch, check the error terms on the entire training set
        print_and_write("validation set errors:", console_log)
        print_and_write("\tclassification error: {:.6f}".format(valid_ce), console_log)
        print_and_write("\tautoencoder error: {:.6f}".format(valid_ae), console_log)
        print_and_write("\terror_1: {:.6f}".format(valid_e1), console_log)
        print_and_write("\terror_2: {:.6f}".format(valid_e2), console_log)
        print_and_write("\ttotal error: {:.6f}".format(valid_te), console_log)
        print_and_write("\taccuracy: {:.4f}".format(valid_ac), console_log)

        # test set accuracy evaluation
        if epoch % test_display_step == 0 or epoch == training_epochs - 1:
            test_ac = 0.0
            for i in range(n_test_batch):
                batch_x, batch_y = mnist.test.next_batch(batch_size)
                ac = sess.run(accuracy,
                              feed_dict={X: batch_x,
                                         Y: batch_y})
                test_ac += ac/n_test_batch

            # after every epoch, check the error terms on the entire training set
            print_and_write("test set:", console_log)
            print_and_write("\taccuracy: {:.4f}".format(test_ac), console_log)

        if epoch % save_step == 0 or epoch == training_epochs - 1:
            # one .meta file is enough to recover the computational graph
            saver.save(sess, os.path.join(model_folder, model_filename),
                       global_step=epoch,
                       write_meta_graph=(epoch == 0 or epoch == training_epochs - 1))

            prototype_imgs = sess.run(X_decoded,
                                      feed_dict={feature_vectors: prototype_feature_vectors.eval()})
            # visualize the prototype images
            n_cols = 5
            n_rows = n_prototypes // n_cols + 1 if n_prototypes % n_cols != 0 else n_prototypes // n_cols
            g, b = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
            for i in range(n_rows):
                for j in range(n_cols):
                    if i*n_cols + j < n_prototypes:
                        b[i][j].imshow(prototype_imgs[i*n_cols + j].reshape(input_height, input_width),
                                        cmap='gray',
                                        interpolation='none')
                        b[i][j].axis('off')
                        
            plt.savefig(os.path.join(img_folder, 'prototype_result-' + str(epoch) + '.png'),
                        transparent=True,
                        bbox_inches='tight',
                        pad_inches=0)
            plt.close()

            # Applying encoding and decoding over a small subset of the training set
            examples_to_show = 10
            encode_decode = sess.run(X_decoded,
                                     feed_dict={X: mnist.train.images[:examples_to_show]})

            # Compare original images with their reconstructions
            f, a = plt.subplots(2, examples_to_show, figsize=(examples_to_show, 2))
            for i in range(examples_to_show):
                a[0][i].imshow(mnist.train.images[i].reshape(input_height, input_width),
                                cmap='gray',
                                interpolation='none')
                a[0][i].axis('off')
                a[1][i].imshow(encode_decode[i].reshape(input_height, input_width), 
                                cmap='gray',
                                interpolation='none')
                a[1][i].axis('off')
                
            plt.savefig(os.path.join(img_folder, 'decoding_result-' + str(epoch) + '.png'),
                        transparent=True,
                        bbox_inches='tight',
                        pad_inches=0)
            plt.close()
    print_and_write("Optimization Finished!", console_log)
console_log.close()