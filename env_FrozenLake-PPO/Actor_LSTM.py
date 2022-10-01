import tensorflow as tf
import numpy as np

class ActorLSTM(tf.keras.Model):
    
    def __init__(self, number_actions: int):

        super(ActorLSTM, self).__init__()
        
        self.conv_one = tf.keras.layers.Conv3D(filters = 16, kernel_size=3, activation=tf.nn.leaky_relu, padding= 'same')
        self.batch_one = tf.keras.layers.BatchNormalization()
        self.activation_one = tf.keras.layers.Activation(tf.nn.leaky_relu)
        self.flat = tf.keras.layers.Flatten()
        self.denseone = tf.keras.layers.Dense(5, activation = tf.nn.leaky_relu)
        self.lstm = tf.keras.layers.LSTM(16, activation = tf.nn.leaky_relu)
        self.densetwo = tf.keras.layers.Dense(32, activation = tf.nn.leaky_relu)
        self.outputlayer = tf.keras.layers.Dense(number_actions, activation = tf.nn.softmax)
        

    def call(self, input):
        #print(input.shape)
        x = self.conv_one(input)
        x = self.batch_one(x)
        x = self.activation_one(x)
        x = self.flat(x)
        x = tf.expand_dims(x, 0)
        x = self.denseone(x)
        x = self.lstm(x)
        x = self.densetwo(x)
        x = self.outputlayer(x)
        return x