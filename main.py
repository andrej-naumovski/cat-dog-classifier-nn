import image_utils
import time
import numpy as np
import nn.config
import nn.nn
import tensorflow as tf


def print_test_accuracy(test_data_images, test_data_labels, test_data_classes, predicted_classes):
    test_batch_size = 256
    num_test = len(test_data_images)
    print(num_test)
    prediction_classes = np.zeros(shape=num_test, dtype=np.int)
    i = 0
    while i < num_test:
        j = min(i + test_batch_size, num_test)
        print(i, j)
        images_batch = test_data_images[i:j, :]
        labels_batch = test_data_labels[i:j, :]

        feed_dict_test = {
            x: images_batch,
            y_true: labels_batch
        }

        prediction_classes[i:j] = session.run(predicted_classes, feed_dict=feed_dict_test)

        i = j

    class_true = test_data_classes

    correct = (class_true == prediction_classes)
    correct_sum = correct.sum()

    test_accuracy = float(correct_sum) / num_test

    print('Test accuracy', test_accuracy)


time_to_process = time.time()

images, labels = image_utils.load_dataset('annotations/trainval.txt')
images_test, labels_test = image_utils.load_dataset('annotations/test.txt', test_data=True)
print(images[0].shape)
classes = np.argmax(labels, axis=1)
classes_test = np.argmax(labels_test, axis=1)

x = tf.placeholder(tf.float32, shape=[None, nn.config.img_size_flat])

x_image = tf.reshape(x, [-1, nn.config.img_width, nn.config.img_height, nn.config.num_channels])

y_true = tf.placeholder(tf.float32, [None, nn.config.num_classes])

y_true_cls = tf.argmax(y_true, axis=1)

conv_layer1, weights1 = nn.nn.generate_conv_layer(x_image, num_channels=nn.config.num_channels,
                                                  layer_data=nn.config.layer1, use_pooling=False)

conv_layer2, weights2 = nn.nn.generate_conv_layer(conv_layer1, num_channels=nn.config.layer1['num_filters'],
                                                  layer_data=nn.config.layer2, use_relu=False)

conv_layer3, weights3 = nn.nn.generate_conv_layer(conv_layer2, num_channels=nn.config.layer2['num_filters'],
                                                  layer_data=nn.config.layer3, use_relu=False)

conv_layer4, weights4 = nn.nn.generate_conv_layer(conv_layer3, num_channels=nn.config.layer3['num_filters'],
                                                  layer_data=nn.config.layer4, use_relu=False, use_pooling=False)

conv_layer5, weights5 = nn.nn.generate_conv_layer(conv_layer4, num_channels=nn.config.layer4['num_filters'],
                                                  layer_data=nn.config.layer5, use_relu=False)

conv_layer6, weights6 = nn.nn.generate_conv_layer(conv_layer5, num_channels=nn.config.layer5['num_filters'],
                                                  layer_data=nn.config.layer6, use_relu=False, use_pooling=False)

conv_layer7, weights7 = nn.nn.generate_conv_layer(conv_layer6, num_channels=nn.config.layer6['num_filters'],
                                                  layer_data=nn.config.layer7, use_relu=False)

conv_layer8, weights8 = nn.nn.generate_conv_layer(conv_layer7, num_channels=nn.config.layer7['num_filters'],
                                                  layer_data=nn.config.layer8, use_relu=False, use_pooling=False)

layer_flat, num_features = nn.nn.flatten_layer(conv_layer6)

layer_fc1 = nn.nn.generate_fc_layer(layer_flat, num_inputs=num_features, num_outputs=nn.config.fully_connected_size,
                                    use_relu=True)

layer_fc2 = nn.nn.generate_fc_layer(layer_fc1, num_inputs=nn.config.fully_connected_size,
                                    num_outputs=nn.config.num_classes, use_relu=True)

y_prediction = tf.nn.softmax(layer_fc2)
y_prediction_class = tf.argmax(y_prediction, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_prediction_class, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()

session.run(tf.global_variables_initializer())

train_batch_size = 64

total_iterations = 0
num_iterations = 1000

for i in range(num_iterations):
    batch = image_utils.generate_batch(images, labels, train_batch_size)
    feed_dict_train = {
        x: batch['images'],
        y_true: batch['labels']
    }
    session.run(optimizer, feed_dict=feed_dict_train)
    if i % 100 == 0 or i == num_iterations - 1:
        acc = session.run(accuracy, feed_dict=feed_dict_train)
        print('Accuracy', acc)

print_test_accuracy(test_data_images=images_test, test_data_labels=labels_test, test_data_classes=classes_test,
                    predicted_classes=y_prediction_class)
