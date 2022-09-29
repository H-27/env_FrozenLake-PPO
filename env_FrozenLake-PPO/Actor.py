import tensorflow as tf
import numpy as np

class Actor(tf.keras.Model):
    
    def __init__(self, number_actions: int, size_first_layer: int, size_second_layer: int, size_third_layer: int = 32):

        super(Actor, self).__init__()
        
        self.conv_one = tf.keras.layers.Conv3D(filters = size_first_layer, kernel_size=1, activation=tf.nn.relu, padding= 'valid')
        self.batch_one = tf.keras.layers.BatchNormalization()
        self.activation_one = tf.keras.layers.Activation(tf.nn.relu)
        # second
        self.conv_two = tf.keras.layers.Conv3D(filters = size_second_layer, kernel_size=3, activation=tf.nn.relu, padding= 'same')
        self.batch_two = tf.keras.layers.BatchNormalization()
        self.activation_two = tf.keras.layers.Activation(tf.nn.relu)
        self.dense = tf.keras.layers.Dense(size_third_layer, activation = tf.nn.relu)
        self.flat = tf.keras.layers.Flatten()
        self.outputlayer = tf.keras.layers.Dense(number_actions, activation = tf.nn.softmax)
        

    def call(self, input):

        x = self.conv_one(input)
        x = self.batch_one(x)
        x = self.activation_one(x)
        x = self.conv_two(x)
        x = self.batch_two(x)
        x = self.activation_two(x)
        #x = self.conc([x, input])
        x = self.dense(x)
        x = self.flat(x)
        x = self.outputlayer(x)
        

        return x