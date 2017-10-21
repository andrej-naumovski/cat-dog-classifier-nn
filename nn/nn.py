import tensorflow as tf
import numpy as np
import nn.config as config


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def generate_conv_layer(input, num_channels, layer_data, use_pooling=True):
    shape = [layer_data['filter_size'], layer_data['filter_size'], num_channels, layer_data['num_filters']]

    weights = new_weights(shape)
    biases = new_biases(layer_data['num_filters'])

    layer = tf.nn.conv2d(input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

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


def print_test_accuracy(test_data_images, test_data_labels, test_data_classes, predicted_classes, session):
    test_batch_size = 256
    num_test = len(test_data_images)
    print(num_test)
    prediction_classes = np.zeros(shape=num_test, dtype=np.int)

    i = 0
    while i < num_test:
        j = min(i + test_batch_size, num_test)
        print(i, j)
        images = test_data_images[i:j, :]
        labels = test_data_labels[i:j, :]

        feed_dict_test = {
            'x': images,
            'y_true': labels
        }

        prediction_classes[i:j] = session.run(predicted_classes, feed_dict=feed_dict_test)

        i = j

    class_true = test_data_classes

    correct = (class_true == prediction_classes)
    correct_sum = correct.sum()

    test_accuracy = float(correct_sum) / num_test

    print('Test accuracy', test_accuracy)