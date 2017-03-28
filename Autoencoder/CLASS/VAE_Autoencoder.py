from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import sklearn.preprocessing as prep

from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import imsave

def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                              minval = low,
                              maxval = high,
                              dtype = tf.float32,
                              seed = 33)

class VariationalAutoencoder(object):

    def __init__(self, n_input, n_hidden, optimizer = tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.z_mean = tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1'])
        self.z_log_sigma_sq = tf.add(tf.matmul(self.x, self.weights['log_sigma_w1']), self.weights['log_sigma_b1'])

        # sample from gaussian distribution
        eps = tf.random_normal(tf.stack([tf.shape(self.x)[0], self.n_hidden]), 0, 1, dtype = tf.float32)
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        self.reconstruction = tf.add(tf.matmul(self.z, self.weights['w2']), self.weights['b2'])

        # cost
        reconstr_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['log_sigma_w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['log_sigma_b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X})

    def transform(self, X):
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.z_mean: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])

def VAE_main():
    print('starting...')
    print('loading data,please wait moment...')
    mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

    def min_max_scale(X_train, X_test):
        preprocessor = prep.MinMaxScaler().fit(X_train)
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)
        return X_train, X_test


    def get_random_block_from_data(data, batch_size):
        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index:(start_index + batch_size)]


    X_train, X_test = min_max_scale(mnist.train.images, mnist.test.images)

    n_samples = int(mnist.train.num_examples)
    training_epochs = 20
    batch_size = 128
    display_step = 1

    autoencoder = VariationalAutoencoder(n_input = 784,
                                         n_hidden = 200,
                                         optimizer = tf.train.AdamOptimizer(learning_rate = 0.001))

    if os.path.exists('result_VAE'):
        os.rename('result_VAE','result_VAE_before')
        path = os.getcwd()
        print(path)
        paths = path + str('\\result_VAE')
        print(paths)
        os.chdir(paths)
        print(os.getcwd())
    else:
        os.mkdir('result_VAE')
        path = os.getcwd()
        print(path)
        paths = path + str('\\result_VAE')
        print(paths)
        os.chdir(paths)
        print(os.getcwd())


    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)

            # Fit training using batch data
            cost = autoencoder.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size
            weights = autoencoder.getWeights
            bias = autoencoder.getBiases
            #data.append(batch_data)
            reconstract = autoencoder.reconstruct(batch_xs) 
            picture = np.reshape(reconstract, [128, 28, 28, -1])
            #print(picture.shape)
            result = picture[1:2]
            #print(result.shape)
            data = np.reshape(result, [28, 28])
            imsave('%d.jpg' %(i), data)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch + 1),  "cost=", "{:.9f}".format(avg_cost))

    print ("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
    print('weights is:', weights)
    print('bias is:', bias)
    print(reconstract.shape)
    print('recontruct result is:', reconstract)
    plt.plot(data)
    plt.show()  
    print('ending...')

if __name__ == '__main__':
    VAE_main()