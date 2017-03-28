# import the packages
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep

from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from scipy.misc import imsave

def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                              minval = low,
                              maxval = high,
                              dtype = tf.float32,
                              seed = 33)

class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus, 
                 optimizer = tf.train.AdamOptimizer(), scale = 0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initial_weights()
        self.weights = network_weights
        
        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),
                                                     self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        
        # cost 
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
    def _initial_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
        return all_weights
    
    def once_fit(self, X):
        cost, _ = self.sess.run((self.cost, self.optimizer), feed_dict = {self.x: X, self.scale: self.training_scale})
        return cost
    
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X, self.scale: self.training_scale})
    
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict = {self.x: X, self.scale: self.training_scale})
    
    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})
    
    def reconstraion(self, X):
        return self.sess.run(self.reconstruction, feed_dict = {self.x: X, self.scale: self.training_scale})
    
    def get_weights(self):
        return self.sess.run(self.weights['w1'])
    
    def get_biases(self):
        return self.sess.run(self.weights['b1'])
    
def AGN_main():
    print('starting...')
    print('loading data,please wait a moment...')
    mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

    def standard_scale(X_train, X_test):
        preprocess = prep.StandardScaler().fit(X_train)
        X_train = preprocess.transform(X_train)
        X_test = preprocess.transform(X_test)
        return X_train, X_test

    def get_batch_data(data, batch_size):
        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index : start_index + batch_size]


    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

    n_samples = int(mnist.train.num_examples)
    nb_epoch = 20
    batch_size = 128
    display_time = 1

    autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784,
                                                   n_hidden = 200,
                                                   transfer_function = tf.nn.relu,
                                                   optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
                                                   scale = 0.01)

    if os.path.exists(os.path.dirname('result')):
        os.rename('result','result_before')
        path = os.getcwd()
        print(path)
        paths = path + str('\\result')
        print(paths)
        os.chdir(paths)
        print(os.getcwd())
    else:
        os.mkdir('result')
        path = os.getcwd()
        print(path)
        paths = path + str('\\result')
        print(paths)
        os.chdir(paths)
        print(os.getcwd())

    for epoch in range(nb_epoch):
        total_batch = int(n_samples / batch_size)
        avg_cost = 0.
        for i in range(total_batch):
            batch_data = get_batch_data(X_train, batch_size)
            cost = autoencoder.once_fit(batch_data)
            avg_cost += cost / n_samples * batch_size
            weights = autoencoder.get_weights
            bias = autoencoder.get_biases
            reconstract = autoencoder.reconstraion(batch_data) 
            picture = np.reshape(reconstract, [128, 28, 28, -1])
            #print(picture.shape)
            result = picture[1:2]
            #print(result.shape)
            data = np.reshape(result, [28, 28])
            imsave('%d.jpg' %(i), data)
        
        if epoch % display_time == 0:
            print('Epoch:', '%04d' %(epoch + 1), 'cost =','{:.9f}'.format(avg_cost))
    print('Total cost is: ' + str(autoencoder.calc_total_cost(X_test)))
    print('weights is:', weights)
    print('bias is:', bias)
    print(reconstract.shape)
    print('recontruct result is:', reconstract)
    plt.plot(data)
    plt.show() 
    print('ending...')
    
if __name__ == '__main__':
    AGN_main()