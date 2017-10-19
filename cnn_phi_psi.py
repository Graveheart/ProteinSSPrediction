from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import os.path
import math

import tensorflow as tf
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

LOGDIR = "/tmp/cnn_backbone_angles/"

# Parameters
batch_size = 5
training_epochs = 10
display_step = 1
window_size = 11
internal_channels_1 = 100
internal_channels_2 = 100
internal_channels_3 = 100
internal_channels_4 = 50

test_buffer = 5
beta = 0.001
keep_prob = tf.placeholder("float")
values_to_predict = 2
num_splits = 5
alpha = 0.2
learning_rate = 1E-3

def fc_layer(input, size_in, size_out, name="fc"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    act = tf.matmul(input, w) + b
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return act


def convnn(x, channels_num, layers_num, window_size = 11):
    W_arr = []
    # First convolutional layer
    input_dimensions = x.get_shape().as_list()[1:]
    filter_shape = [window_size, input_dimensions[-1], channels_num]
    W_input = weight_variable(filter_shape)
    W_arr.append(W_input)
    b_input = bias_variable([input_dimensions[0], channels_num])
    layers = []

    layers.append(tf.nn.relu(conv1d(x, W_input) + b_input))

    # Hidden layers
    filter_shape = [window_size, channels_num, channels_num]
    W_hidden = tf.constant([], dtype=tf.float32)
    for i in range(layers_num):
        with tf.name_scope("conv"):
            W_hidden = weight_variable(filter_shape)
            W_arr.append(W_hidden)
            b_hidden = bias_variable([input_dimensions[0], channels_num])
            conv_layer = tf.nn.tanh(alpha*conv1d(layers[i], W_hidden) + b_hidden)
            tf.summary.histogram("weights", W_hidden)
            tf.summary.histogram("biases", b_hidden)
            tf.summary.histogram("activations", conv_layer)
        with tf.name_scope("dropout"):
            keep_prob = tf.placeholder_with_default(1.0, shape=(), name="keep_prob")
            dropout = tf.layers.dropout(conv_layer, keep_prob)
            layers.append(dropout)
    # x_reshape = tf.reshape(x, [-1, input_dimension])

    output_shape = 50
    # Output convolutional layer
    # filter_shape1 = [window_size, channels_num, output_shape]
    # W_output1 = weight_variable(filter_shape)
    # W_arr.append(W_output1)
    # b_output1 = bias_variable([input_dimensions[0], channels_num])
    # layer_out = conv1d(layers[-1], W_output1) + b_output1
    # layer_out =  tf.atan2(tf.sin(layer_out), tf.cos(layer_out))

    filter_shape = [window_size, channels_num, values_to_predict]
    W_output = weight_variable(filter_shape)
    W_arr.append(W_output)
    b_output = bias_variable([values_to_predict])
    layer_out = conv1d(layers[-1], W_output) + b_output

    # Loss function with L2 Regularization with beta=0.001
    print()
    regularizers = tf.nn.l2_loss(W_input) + tf.nn.l2_loss(W_hidden) * layers_num + tf.nn.l2_loss(W_output)

    # regularizers = tf.constant(0, dtype=tf.float32)
    # for W in W_arr:
    #     regularizers += tf.nn.l2_loss(W)

    return layer_out, regularizers

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name="W")


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name="B")

def conv1d(x, W):
    """conv1d returns a 1d convolution layer."""
    return tf.nn.conv1d(x, W, 1, 'SAME')

def avgpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def calculate_accuracy(predictions, labels, ):
    num_proteins = predictions.shape[0]
    protein_accuracy = np.zeros(num_proteins, dtype=np.float32)

    for i in range(num_proteins):
        total_predictions = 0
        correct_predictions = 0
        for j in range(predictions.shape[1]):
            phi = labels[i][j][0]
            psi = labels[i][j][1]
            if ((phi != 0) & (psi != 0)):
                total_predictions += 1
                expected_state = get_backbone_distribution(labels[i][j])
                predicted_state = get_backbone_distribution(predictions[i][j])
                if (predicted_state == expected_state):
                    correct_predictions += 1
                # print("REAL PHI->>>>>"+str(labels[i][j][0]))
                # print("PREDICTED PHI->>>>>" + str(predictions[i][j][0]))
                diff = math.sqrt(math.pow(phi - predictions[i][j][0], 2)+math.pow(psi - predictions[i][j][1], 2))
                # diff_phi = labels[i][j][0] - predictions[i][j][0]
                # diff_psi = labels[i][j][1] - predictions[i][j][1]
                # criteria_1 = np.abs(diff_phi) < 60 || np.abs(diff_psi) < 60
                # if (diff < math.pi/3):
                #     correct_predictions += 1
        if (total_predictions > 0):
            # print("CORRECT->>>>>"+str(correct_predictions))
            # print("TOTAL->>>>>" + str(total_predictions))
            protein_accuracy[i] = correct_predictions / float(total_predictions)
            # if (protein_accuracy < 1):
            #     print(protein_accuracy)
    # print(protein_accuracy)
    # accuracy = 0
    # if (total_predictions > 0):
    #     accuracy = sum(correct_predictions) / total_predictions
    return protein_accuracy

def get_backbone_distribution(angles):
    phi = math.degrees(angles[0])
    psi = math.degrees(angles[1])
    #  A: -160 < phi <0 and -70 < psi < 60
    if ((-160 < phi < 0) & (-70 < psi < 60)):
        return 1
    # P: 0 < phi < 160 and -60 < psi < 95
    elif ((0 < phi < 160) & (-60 < psi < 95)):
        return 2
    else:
        return 3

def plot_ramachandran(predictions, title):
    phi = predictions[:][:][0].flatten()
    psi = predictions[:][:][1].flatten()
    colors = np.random.rand(len(psi))
    f = plt.figure()
    plt.xlim([-math.pi, math.pi])
    plt.ylim([-math.pi, math.pi])
    plt.title(title)
    plt.xlabel('phi')
    plt.ylabel('psi')
    plt.grid()
    plt.scatter(phi, psi, alpha=0.5, c=colors)
    f.show()

def plot_loss(loss_arr):
    l = plt.figure()
    plt.plot(loss_arr)
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(plot_legend, loc='upper left')
    l.show()

def make_hparam_string(layers_num, channels_num, test_session):
  return "nl_%s,nc_%s, session%s" % (layers_num, channels_num, test_session)


data = np.load('phipsi_features.npz')['features']
all_data = data.reshape(data.shape[0],700,69)
# all_data = all_data[0:300]
all_sets = all_data[:,:,0:21]
all_sets = np.concatenate([all_sets, all_data[:,:,21:42]], axis=-1)
# all_sets = np.concatenate([all_sets, all_data[:,:,42:63]], axis=-1)
# all_labels = all_data[:,:,63:67]
all_angles = all_data[:,:,67:69]
where_are_NaNs = np.isnan(all_angles)
all_angles[where_are_NaNs] = 0
k_fold = KFold(n_splits=num_splits)

layers_channels = [(3, 100)]
# Build the convolutional network
for layers_num, channels_num in layers_channels:
    for dropout_rate in [0.25]:
        crossvalidation_train_accuracy = 0
        crossvalidation_test_accuracy = 0
        executed_epochs = 0
        train_session = 0
        test_session = 0
        learning_rate_type = 1
        for train_index, test_index in k_fold.split(all_sets):
            train_set, test_set = all_sets[train_index], all_sets[test_index]
            train_labels, test_labels = all_angles[train_index], all_angles[test_index]
            train_size = train_set.shape[0]
            train_y = train_labels
            test_y = test_labels
            test_session += 1

            # Create the model
            x = tf.placeholder(tf.float32, [None, 700, train_set[0].shape[-1]], name="x")

            # Define loss and optimizer
            y_ = tf.placeholder(tf.float32, [None, 700, values_to_predict], name="labels")

            y_nn, regularizers = convnn(x, channels_num, layers_num, window_size)
            prediction = y_nn

            with tf.name_scope("loss"):
                deviations = tf.subtract(prediction, y_)
                atan2 = tf.atan2(tf.sin(deviations), tf.cos(deviations))
                loss = tf.square(atan2, name="loss")
                mean_loss = tf.reduce_mean(loss)
                loss_summary = tf.summary.scalar("loss", mean_loss)

            # with tf.name_scope("loss2"):
            #     print(tf.shape(prediction))
            #     print(tf.shape(y_))
            #     phi = prediction[:, :, 0]
            #     phi0 = y_[:, :, 0]
            #     psi = prediction[:, :, 1]
            #     psi0 = y_[:,:, 1]
            #     cos_phi = tf.square(tf.cos(tf.subtract(phi, phi0)))
            #     sin_phi = tf.square(tf.sin(tf.subtract(phi, phi0)))
            #     cos_psi = tf.square(tf.cos(tf.subtract(psi, psi0)))
            #     sin_psi = tf.square(tf.sin(tf.subtract(psi, psi0)))
            #     phi_squared_sum = tf.add(cos_phi, sin_phi)
            #     psi_squared_sum = tf.add(cos_psi, sin_psi)
            #     loss2 = tf.add(phi_squared_sum, psi_squared_sum)

            with tf.name_scope("mse"):
                mse = tf.reduce_mean(tf.squared_difference(prediction, y_))
                mse_summary = tf.summary.scalar("mse", mse)

            with tf.name_scope("l2_loss"):
                l2_loss = beta * regularizers
                loss = tf.reduce_mean(loss + l2_loss)
                l2_summary = tf.summary.scalar("l2_loss", l2_loss)

            with tf.name_scope("train"):
                # Use Adam optimizer
                optimization = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            # with tf.name_scope("accuracy"):
            #     correct_prediction = tf.equal(prediction, y)
            #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            #     tf.summary.scalar("accuracy", accuracy)

            summ = tf.summary.merge_all()

            print("Window size: " + str(window_size))
            print("Layers: " + str(layers_num))
            print("Channels: " +  str(channels_num))
            print("Beta: " + str(beta))
            print("Dropout rate: " + str(dropout_rate))
            sess = tf.InteractiveSession()
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver()

            previous_loss = 0.0
            plot_legend = []
            for epoch in range(training_epochs):
                train_session += 1
                hparam = make_hparam_string(layers_num, channels_num, train_session)
                writer = tf.summary.FileWriter(LOGDIR + hparam)
                writer.add_graph(sess.graph)
                total_batches = int(train_size/batch_size)
                loss_arr = []
                # Loop over all batches
                for i in range(total_batches):
                    start_index = i * batch_size
                    stop_index = (i+1) * batch_size
                    batch_x = train_set[start_index:stop_index]
                    batch_y = train_y[start_index:stop_index]
                    # Run optimization op
                    # backprop and cost op (to get loss value)
                    if i % 5 == 0:
                        batch_predictions, l_summ, batch_loss = sess.run([prediction, loss_summary, loss], feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
                        writer.add_summary(l_summ, i+1)
                        loss_arr.append(batch_loss)
                        saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
                        batch_accuracy = np.average(calculate_accuracy(batch_predictions, batch_y))
                        # print('step %d, training accuracy %g' % (i, batch_accuracy))
                    optimization.run(feed_dict={x: batch_x, y_: batch_y})

                # Display logs per epoch step
                if epoch % display_step == 0:
                    predictions, train_mse = sess.run([prediction,mse], feed_dict={x: train_set, y_: train_y, keep_prob: dropout_rate})
                    # plot_ramachandran(predictions, "Predictions")
                    # plot_ramachandran(train_y, "Real values")
                    # plot_loss(loss_arr)
                    # raw_input()
                    train_accuracy = calculate_accuracy(predictions, train_y)
                    train_accuracy=np.average(train_accuracy)
                    crossvalidation_train_accuracy += train_accuracy
                    plot_legend.append('train_' + str(epoch))

                    print("Training accuracy: ", \
                      "{:.6f}".format(train_accuracy))
                executed_epochs += 1
            # Test trained model
            test_predictions, test_summ, test_mse = sess.run([prediction, loss_summary, mse], feed_dict={x: test_set, y_: test_y})
            writer.add_summary(test_summ, i + 1)
            test_accuracy = calculate_accuracy(test_predictions, test_y)
            # plt.plot(test_accuracy)

            # plot_legend.append('validation')


            test_accuracy = np.average(test_accuracy)
            crossvalidation_test_accuracy += test_accuracy
            print("Testing accuracy: ", \
                  "{:.6f}".format(test_accuracy))
            print("Testing MSE: ", \
                  "{:.6f}".format(test_mse))
        print("Final Training accuracy: ", \
              "{:.6f}".format(crossvalidation_train_accuracy / (num_splits*training_epochs)))
        print("Final Test accuracy: ", \
                    "{:.6f}".format(crossvalidation_test_accuracy / num_splits))
        print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)

        # valid_predictions = sess.run(tf.argmax(prediction, 2), feed_dict={x: valid_x, y_: valid_y})
        # valid_labels = np.argmax(valid_y, 2)
        # valid_accuracy = calculate_accuracy(valid_predictions, valid_labels)
        # print("Validation accuracy: ", \
        #             "{:.6f}".format(valid_accuracy))