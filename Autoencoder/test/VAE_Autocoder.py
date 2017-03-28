# import the packages
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
import time
import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep

from CLASS.CLASS_VAE import *
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from scipy.misc import imsave

flags = tf.app.flags
flags.DEFINE_integer('nb_epochs', 2, 'the numbers of the epoch')
flags.DEFINE_integer('batch_size', 128, 'the size of the batch')
flags.DEFINE_integer('display_time', 1, 'the time of the display')
flags.DEFINE_float('learning_rate', 0.001, 'the learning rate of the optimizer')
flags.DEFINE_string('your_path', 'D:/Data Minning/train_code/train/Autoencoder/test', 'the path of you code')
flags.DEFINE_string('optimizer', 'adag', 'choose the right optimizer')
FLAGS = flags.FLAGS

def standard_scale(X_train, X_test):
    preprocess = prep.StandardScaler().fit(X_train)
    X_train = preprocess.transform(X_train)
    X_test = preprocess.transform(X_test)
    return X_train, X_test

def get_batch_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index : start_index + batch_size]
    
def Save_Result():
    if os.path.exists(os.path.dirname('result_VAE')):
        os.rename('result_VAE','result_VAE_before')
        os.mkdir('result_VAE')
        path = os.getcwd()
        #print(path)
        paths = path + str('\\result_VAE')
        #print(paths)
        os.chdir(paths)
        #print(os.getcwd())
    else:
        os.mkdir('result_VAE')
        path = os.getcwd()
        #print(path)
        paths = path + str('\\result_VAE')
        #print(paths)
        os.chdir(paths)
        #print(os.getcwd())
        
def Save_Origial():
    if os.path.exists(os.path.dirname('origial_VAE')):
        os.rename('origial_VAE','origial_before_VAE')
        path = os.getcwd()
        #print(path)
        paths = path + str('\\origial_VAE')
        #print(paths)
        os.chdir(paths)
        #print(os.getcwd())
    else:
        os.mkdir('origial_VAE')
        path = os.getcwd()
        #print(path)
        paths = path + str('\\origial_VAE')
        #print(paths)
        os.chdir(paths)
        #print(os.getcwd())
        
def Save_transform():
    if os.path.exists(os.path.dirname('transform_VAE')):
        os.rename('transform_VAE','transform_before_VAE')
        path = os.getcwd()
        #print(path)
        paths = path + str('\\transform_VAE')
        #print(paths)
        os.chdir(paths)
        #print(os.getcwd())
    else:
        os.mkdir('transform_VAE')
        path = os.getcwd()
        #print(path)
        paths = path + str('\\transform_VAE')
        #print(paths)
        os.chdir(paths)
        #print(os.getcwd())
        
def choose_optimizer(name):
    if name == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    elif name == 'adam':
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    elif name == 'adag':
        optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate)
    elif name == 'adad':
        optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate)
    elif name == 'rmsp':
        optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
    else:
        print('please add you optimizer...')
        raise Exception('Error...')
    return optimizer
    
def print_information(cost, epoch):
    plt.xlabel('the number of each epoch')
    plt.ylabel('the average cost of each epoch')
    plt.title('the picture of the cost')
    plt.plot(epoch, cost)
    plt.show() 
    print('ending...')
    
#def main(unused_argv):
def main(_):
    start_time = time.time()
    print('starting...')
    print('loding data,please wait a moment...')
    #print('\n')
    
    mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
    n_samples = int(mnist.train.num_examples)
    
    # load the mnist datasets and print the shape
    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
    print(mnist.train.images.shape)
    print(X_train.shape)
    print(X_test.shape)
    #print('\n')
    
    # Instance an object
    autoencoder = VariationalAutoencoder(n_input = 784,
                                         n_hidden = 256,
                                         optimizer = choose_optimizer(name = FLAGS.optimizer))
    # save the origial pictures
    Save_Origial()
    for epoch1 in range(FLAGS.nb_epochs):
        total_batch = int(n_samples / FLAGS.batch_size)
        for i in range(total_batch):
            batch_data = get_batch_data(X_train, FLAGS.batch_size)
            origial = np.reshape(batch_data, [128, 28, 28, -1])
            origial_picture = origial[1:2]
            origial_result = np.reshape(origial_picture, [28, 28])
            imsave('%d.jpg' %(i), origial_result)
    # get back to the upper path 
    path = FLAGS.your_path
    print('start saving the origial pictures...')
    print(path)
    os.chdir(path)
    
    # save the result of the hidden layer
    Save_transform()
    for epoch1 in range(FLAGS.nb_epochs):
        total_batch = int(n_samples / FLAGS.batch_size)
        for j in range(total_batch):
            batch_data = get_batch_data(X_train, FLAGS.batch_size)
            transforms = autoencoder.transform(batch_data)
            #print(transforms.shape)
            transform = np.reshape(transforms, [128, 16, 16, -1])
            transform_picture = transform[1:2]
            transform_result = np.reshape(transform_picture, [16, 16])
            imsave('%d.jpg' %(j), transform_result)
    # get back to the upper path 
    path = FLAGS.your_path
    print('start saving the hidden layers pictures...')
    print(path)
    os.chdir(path)
          
    # save the reconstraction pictures    
    Save_Result()
    cost_value = []
    epochs = []
    for epoch in range(FLAGS.nb_epochs):
        total_batch = int(n_samples / FLAGS.batch_size)
        avg_cost = 0.
        for k in range(total_batch):
            batch_data = get_batch_data(X_train, FLAGS.batch_size)
            cost = autoencoder.partial_fit(batch_data)
            avg_cost += cost / n_samples * FLAGS.batch_size
            reconstract = autoencoder.reconstruct(batch_data) 
            picture = np.reshape(reconstract, [128, 28, 28, -1])
            result = picture[1:2]
            data = np.reshape(result, [28, 28])
            imsave('%d.jpg' %(k), data)
            
        cost_value.append(avg_cost)
        epochs.append(epoch)
        
        if epoch % FLAGS.display_time == 0:
            print('Epoch:', '%04d' %(epoch + 1), 'cost =','{:.9f}'.format(avg_cost))
    print('Total cost is: ' + str(autoencoder.calc_total_cost(X_test)))
    print_information(cost = cost_value, epoch = epochs)
    print('Total time is %d s' %(time.time() - start_time))
    
if __name__ == '__main__':
    tf.app.run()
    #sys.exit(0)
    #tf.app.run(main=None, argv=None)
    #AGN_main()