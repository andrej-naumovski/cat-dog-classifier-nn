import tensorflow as tf
import numpy as np
import nn.config as config


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def generate_conv_layer(input, num_channels, layer_data, use_pooling=True, use_relu=True):
    shape = [layer_data['filter_size'], layer_data['filter_size'], num_channels, layer_data['num_filters']]

    weights = new_weights(shape)
    biases = new_biases(layer_data['num_filters'])

    layer = tf.nn.conv2d(input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer, weights


def flatten_layer(layer):
    layer_shape = layer.get_shape()

    num_features = layer_shape[1:].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def generate_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = new_weights([num_inputs, num_outputs])
    biases = new_biases(num_outputs)

    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer