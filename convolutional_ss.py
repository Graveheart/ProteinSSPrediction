from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import sys

import tensorflow as tf
from sklearn.model_selection import KFold

# Parameters
batch_size = 5
training_epochs = 10
display_step = 1
window_size = 11
internal_channels_1 = 100
internal_channels_2 = 100
internal_channels_3 = 100
internal_channels_4 = 50

train_size = 5600
test_starting_index = 5605
test_size = 272
valid_size = 256
test_buffer = 5
dropout_rate = 0.5 # Dropout, probability to keep units
beta = 0.001
keep_prob = tf.placeholder("float")

def convnn(x, channels_num, layers_num):
    # First convolutional layer
    input_dimensions = x.get_shape().as_list()[1:]
    filter_shape = [window_size, input_dimensions[-1], channels_num]
    W = weight_variable(filter_shape)
    b = bias_variable([input_dimensions[0], channels_num])
    layers = []

    layers.append(tf.nn.relu(conv1d(x, W) + b))

    # Hidden layers
    filter_shape = [window_size, channels_num, channels_num]
    W_hidden = weight_variable(filter_shape)
    b_hidden = bias_variable([input_dimensions[0], channels_num])
    for i in range(layers_num):
        conv_layer = tf.nn.relu(conv1d(layers[i], W_hidden) + b_hidden)
        keep_prob = tf.placeholder_with_default(1.0, shape=())
        dropout = tf.layers.dropout(conv_layer, keep_prob)
        layers.append(dropout)
    # x_reshape = tf.reshape(x, [-1, input_dimension])
    # filter_shape = [window_size, input_dimensions[-1], channels[0]]
    # W_1 = weight_variable(filter_shape)
    # b_1 = bias_variable([input_dimensions[0], channels[0]])
    # layer_1 = tf.nn.relu(conv1d(x, W_1) + b_1)

    # Dropout on hidden layer: RELU layer
    # layer_1_dropout = tf.nn.dropout(layer_1, keep_prob)

    # filter_shape = [window_size, channels[0], channels[1]]
    # W_2 = weight_variable(filter_shape)
    # b_2 = bias_variable([input_dimensions[0], channels[1]])
    # layer_2 = tf.nn.relu(conv1d(layer_1, W_2) + b_2)
    #
    # # Dropout on hidden layer: RELU layer
    # layer_2_dropout = tf.nn.dropout(layer_2, keep_prob)
    #
    # # Third convolutional layer
    # filter_shape = [window_size, channels[1], channels[2]]
    # W_3 = weight_variable(filter_shape)
    # b_3 = bias_variable([input_dimensions[0], channels[2]])
    # layer_3 = tf.nn.relu(conv1d(layer_2, W_3) + b_3)

    # filter_shape = [window_size, channels[2], internal_channels_4]
    # W_4 = weight_variable(filter_shape)
    # b_4 = bias_variable([input_dimensions[0], internal_channels_4])
    # layer_4 = tf.nn.relu(conv1d(layer_3, W_4) + b_4)

    # filter_shape = [window_size, channels[2], internal_channels_4]
    # W_5 = weight_variable(filter_shape)
    # b_5 = bias_variable([input_dimensions[0], internal_channels_4])
    # layer_5 = tf.nn.relu(conv1d(layer_4, W_5) + b_5)

    # logits = tf.layers.dense(inputs=layer_4, units=4)
    #
    # print(logits.shape)

    # # Add dropout operation; 0.6 probability that element will be kept


    # Output convolutional layer
    filter_shape = [window_size, channels_num, 4]
    W_output = weight_variable(filter_shape)
    b_output = bias_variable([4])
    layer_out = conv1d(layers[-1], W_output) + b_output

    # Loss function with L2 Regularization with beta=0.001
    regularizers = tf.nn.l2_loss(W) + layers_num * tf.nn.l2_loss(W_hidden) * layers_num + tf.nn.l2_loss(W_output)

    return layer_out, regularizers

def preprocess(labels, size):
    y = np.zeros((size, 700, 4))
    # Transform 8-state SS into 3-state SS
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            label = labels[i][j]
            # label for strand
            if (label[1] or label[2]):
                y[i][j] = np.array([1., 0., 0., 0.])
            # label for helix
            elif (label[3] or label[4] or label[5]):
                y[i][j] = np.array([0., 1., 0., 0.])
            # label for coil
            elif (label[0] or label[6] or label[7]):
                y[i][j] = np.array([0., 0., 1., 0.])
            else:
                y[i][j] = np.array([0., 0., 0., 1.])
    return y

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv1d(x, W):
    """conv1d returns a 1d convolution layer."""
    return tf.nn.conv1d(x, W, 1, 'SAME')

def avgpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def calculate_accuracy(predictions, labels):
    indices = (labels != 3)
    correct_predictions = (labels == predictions)
    return np.average(correct_predictions[indices])

data = np.load('data/cullpdb.npy')
train_data = data[0:train_size].reshape(train_size,700,57)
train_x = train_data[:,:,0:22]
train_x = np.concatenate([train_x, train_data[:,:,35:57]], axis=-1)
labels_y = train_data[:,:,22:31]
train_y = preprocess(labels_y, train_size)

test_last_index = test_starting_index+test_size
test_data = data[test_starting_index:test_last_index].reshape(test_size,700,57)
test_x = test_data[:,:,0:22]
test_x = np.concatenate([test_x, test_data[:,:,35:57]], axis=-1)
labels_y = test_data[:,:,22:31]
test_y = preprocess(labels_y, test_size)

valid_data = data[test_last_index:test_last_index+valid_size].reshape(valid_size,700,57)
valid_x = valid_data[:,:,0:22]
valid_x = np.concatenate([valid_x, valid_data[:,:,35:57]], axis=-1)
labels_y = valid_data[:,:,22:31]
valid_y = preprocess(labels_y, valid_size)

all_data = data.reshape(data.shape[0],700,57)
all_sets = all_data[:,:,0:22]
all_sets = np.concatenate([all_sets, all_data[:,:,35:57]], axis=-1)
all_labels = all_data[:,:,22:31]
k_fold = KFold(n_splits=5)

layers_channels = [(3, 100)]
beta_arr = [0.02, 0,1, 0.5]
# Build the convolutional network
for layers_num, channels_num in layers_channels:
    for beta in beta_arr:
        crossvalidation_train_accuracy = 0
        crossvalidation_test_accuracy = 0
        executed_epochs = 0
        learning_rate_type = 1
        for train_index, test_index in k_fold.split(all_sets):
            train_set, test_set = all_sets[train_index], all_sets[test_index]
            train_labels, test_labels = all_labels[train_index], all_labels[test_index]
            train_size = train_set.shape[0]
            train_y = preprocess(train_labels, train_size)
            test_y = preprocess(test_labels, test_set.shape[0])

            # Create the model
            x = tf.placeholder(tf.float32, [None, 700, train_set[0].shape[-1]])

            # Define loss and optimizer
            y_ = tf.placeholder(tf.float32, [None, 700, 4])

            y_nn, regularizers = convnn(x, channels_num, layers_num)
            prediction = tf.nn.softmax(y_nn)

            # Normal loss function
            unregularized_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_nn)

            # Loss function with L2 Regularization with beta=0.01
            l2_loss = beta * regularizers

            # We add L2 loss to our unregularized loss
            # loss = tf.reduce_mean(tf.add(unregularized_loss, l2_loss, name='loss'))
            loss = tf.reduce_mean(unregularized_loss + l2_loss)
            # optimization = tf.train.AdamOptimizer()
            # Use Adam optimizer
            # if(learning_rate_type == 0):
            optimization = tf.train.AdamOptimizer().minimize(loss)
            # else:
            #     print("Decaying learning rate")
            #     step = tf.Variable(0, trainable=False)
            #     learning_rate = tf.train.exponential_decay(0.15, step, 1, 0.9999)
            #     optimization = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=step)
            print("Window size: " + str(window_size))
            print("Layers: " + str(layers_num))
            print("Channels: " +  str(channels_num))
            print("Beta: " + str(beta))
            sess = tf.InteractiveSession()
            init = tf.global_variables_initializer()
            sess.run(init)

            training_acc = 0.0
            for epoch in range(training_epochs):
                total_batch = int(train_size/batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    start_index = (i - 1) * batch_size
                    stop_index = i * batch_size
                    batch_x = train_set[start_index:stop_index]
                    batch_y = train_y[start_index:stop_index]
                    # Run optimization op
                    # (backprop) and cost op (to get loss value)
                    if i*batch_size % 400 == 0:
                        predictions = sess.run(tf.argmax(prediction, 2), feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
                        labels = np.argmax(batch_y, 2)
                        batch_accuracy = calculate_accuracy(predictions, labels)
                        # print('step %d, training accuracy %g' % (i, batch_accuracy))
                    optimization.run(feed_dict={x: batch_x, y_: batch_y})

                # Display logs per epoch step
                if epoch % display_step == 0:
                    predictions = sess.run(tf.argmax(prediction, 2), feed_dict={x: train_set, y_: train_y, keep_prob: 0.5})
                    labels = np.argmax(train_y, 2)
                    train_accuracy = calculate_accuracy(predictions, labels)
                    crossvalidation_train_accuracy += train_accuracy
                    print("Training accuracy: ", \
                      "{:.6f}".format(train_accuracy))
                executed_epochs += 1
                if train_accuracy - training_acc < 0.001:
                    break
                else:
                    training_acc = train_accuracy

            # Test trained model
            test_predictions = sess.run(tf.argmax(prediction, 2), feed_dict={x: test_set, y_: test_y})
            labels = np.argmax(test_y, 2)
            crossvalidation_test_accuracy += calculate_accuracy(test_predictions, labels)
        print("Final Training accuracy: ", \
              "{:.6f}".format(crossvalidation_train_accuracy / executed_epochs))
        print("Final Test accuracy: ", \
                    "{:.6f}".format(crossvalidation_test_accuracy / 5))

        # valid_predictions = sess.run(tf.argmax(prediction, 2), feed_dict={x: valid_x, y_: valid_y})
        # valid_labels = np.argmax(valid_y, 2)
        # valid_accuracy = calculate_accuracy(valid_predictions, valid_labels)
        # print("Validation accuracy: ", \
        #             "{:.6f}".format(valid_accuracy))
