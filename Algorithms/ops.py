import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import itertools


#################################################
# Activation
#################################################

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


# various implementations of this exist, particularly in how the negatives are calculated
# for more, read https://arxiv.org/abs/1502.01852
def parametric_relu(X, regularizer=None, name="parametric_relu"):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alphas',
                                 regularizer=regularizer,
                                 dtype=X.dtype,
                                 shape=X.get_shape().as_list()[-1],
                                 initializer=tf.constant_initializer(0.01))
        positives = tf.nn.relu(X)
        negatives = alphas * (tf.subtract(X, tf.abs(X))) * 0.5
        return positives + negatives


# this implementation assumes alpha will be less than 1. It would be stupid if it weren't.
def leaky_relu(X, alpha=0.2):
    return tf.maximum(X, alpha * X)


#################################################
# Network Layers
#################################################

# https://arxiv.org/abs/1302.4389
# This doesn't need a scope. There aren't trainable parameters here. It's just a pool
def maxout(X, num_maxout_units, axis=None):
    input_shape = X.get_shape().as_list()

    axis = -1 if axis is None else axis

    num_filters = input_shape[axis]

    if num_filters % num_maxout_units != 0:
        raise ValueError(
            "num filters (%d) must be divisible by num maxout units (%d)" % (num_filters, num_maxout_units))

    output_shape = input_shape.copy()
    output_shape[axis] = num_maxout_units
    output_shape += [num_filters // num_maxout_units]
    return tf.reduce_max(tf.reshape(X, output_shape), -1, keep_dims=False)


##############
# Dense
##############
def dense(X, shape, w_initializer, name="dense"):
    with tf.variable_scope(name):
        W = tf.get_variable('W_dense', shape=shape, initializer=w_initializer)
        b = tf.get_variable('b_dense', shape=[shape[-1]],
                            initializer=tf.constant_initializer(0.0))
        return tf.matmul(X, W) + b


def dense_layer(inputs, shape, activation, layer_name, use_batch_norm, drop_out, is_train,
                weight_initializer=tf.contrib.layers.xavier_initializer()):
    layer = dense(inputs, shape, weight_initializer, name=layer_name)
    if use_batch_norm:
        layer = tf.layers.batch_normalization(layer, training=is_train)
    layer = activation(layer)
    layer = tf.layers.dropout(layer, rate=drop_out, training=is_train)
    return layer


"""def dense_layer(inputs, weights, biases, activation, use_batch_norm, drop_out, is_train):
    layer = tf.matmul(inputs, weights) + biases
    if use_batch_norm:
        layer = tf.layers.batch_normalization(layer, training=is_train)
    layer = activation(layer)
    layer = tf.layers.dropout(layer, rate=drop_out, training=is_train)
    return layer"""


##############
# Conv
##############
def conv(X,
         output_filter_size,
         kernel=[5, 5],
         strides=[2, 2],
         w_initializer=xavier_initializer(),
         regularizer=None,
         name="conv"):
    with tf.variable_scope(name):
        W = tf.get_variable('W_conv',
                            regularizer=regularizer,
                            dtype=X.dtype,
                            shape=[kernel[0], kernel[1], X.get_shape().as_list()[-1], output_filter_size],
                            initializer=w_initializer)
        b = tf.get_variable('b_conv',
                            regularizer=regularizer,
                            dtype=X.dtype,
                            shape=[output_filter_size],
                            initializer=tf.zeros_initializer(dtype=X.dtype))

        return tf.nn.bias_add(tf.nn.conv2d(X,
                                           W,
                                           strides=[1, strides[0], strides[1], 1],
                                           padding='SAME',
                                           name="conv2d"), b)


def conv_out_size(W, F, P, S):
    # W : input size (width or height)
    # F : filters size
    # P : padding
    # S : stride
    return (W - F + 2 * P) / S + 1


def conv_layer(inputs, filters, kernel, stride, activation, layer_name, use_batch_norm, drop_out,
               is_train, weight_initializer=xavier_initializer()):
    layer = conv(inputs,
                 filters,
                 kernel=[kernel, kernel],
                 strides=[stride, stride],
                 w_initializer=weight_initializer,
                 name=layer_name)
    if use_batch_norm:
        layer = tf.layers.batch_normalization(layer, training=is_train)
    layer = activation(layer)
    layer = tf.layers.dropout(layer, rate=drop_out, training=is_train)
    return layer


def residual_conv_block(inputs, filters, kernel, stride, activation, layer_name, use_batch_norm,
                        drop_out, is_train, weight_initializer=xavier_initializer()):
    layer = conv_layer(inputs, filters, kernel, stride, activation, layer_name + "_1", use_batch_norm, drop_out,
                       is_train, weight_initializer)
    layer = conv_layer(layer, filters, kernel, stride, tf.identity, layer_name + "_2", use_batch_norm, drop_out,
                       is_train, weight_initializer)
    layer += inputs
    layer = activation(layer)
    return layer


#################################################
# Data augmentation and transformation
#################################################

def dirichlet_noise(plane, alpha, epsilon):
    alphas = np.full(plane.shape, alpha)
    out = (1 - epsilon) * plane + epsilon * np.random.dirichlet(alphas)
    return out


def dihedral_transformation(plane, k_rotate, reflection=False):
    new_plane = plane
    if reflection:
        new_plane = np.flip(new_plane, 1)
    new_plane = np.rot90(new_plane, k_rotate)
    return new_plane


def dihedral_transformation_random(plane):
    k_rotate = np.random.randint(low=0, high=4)
    reflection = [True, False][np.random.randint(low=0, high=2)]
    return dihedral_transformation(plane, k_rotate, reflection)


def data_transformation(planes, policy, board_size, k_rotate, reflection):
    num_board_planes = planes.shape[3] - 1  # "- 1" to not include player feature plane

    # Rotate/reflect policy out
    p_pass = policy[0][-1]
    t_p = np.reshape(policy[0][:-1], (board_size, board_size))
    new_p = dihedral_transformation(t_p, k_rotate, reflection)
    new_p = np.append(new_p, p_pass)
    new_p = np.reshape(new_p, (1, board_size * board_size + 1))

    # Rotate/reflect planes
    new_planes = np.copy(planes)
    for i in range(num_board_planes):
        temp_plane = np.reshape(planes[:, :, :, i], (board_size, board_size))
        new_plane = dihedral_transformation(temp_plane, k_rotate, reflection)
        new_plane = np.reshape(new_plane, (1, board_size, board_size))
        new_planes[:, :, :, i] = new_plane

    return new_planes, new_p


# Data augmentation from raw neural network inputs
def data_augmentation_single(planes, policy, board_size, idx=None):
    out_planes = []
    out_policies = []

    all_reflection = [False, True]
    all_rotate = list(range(0, 4))
    all_transformation = list(itertools.product(all_reflection, all_rotate))
    if idx is None:
        reflection_to_do, rotate_to_do = all_reflection, all_rotate
    else:
        reflection_to_do, rotate_to_do = [all_transformation[idx][0]], [all_transformation[idx][1]]

    for reflection in reflection_to_do:
        for k_rotate in rotate_to_do:
            new_planes, new_p = data_transformation(planes, policy, board_size, k_rotate, reflection)

            out_planes.append(new_planes)
            out_policies.append(new_p)

    return out_planes, out_policies


def data_augmentation(states, policies, values, board_size, input_planes, idx=None):
    # states   [N, size, size, no_input_planes]
    # policies [N, size * size + 1]
    # values   [N, 1]
    # idx: value between 0 and 7 or None (choose a particular augmentation or all)

    t_states, t_policies, t_values = reshape_data_for_augmentation(states, policies, values, board_size, input_planes)
    new_states, new_policies, new_values = [], [], []
    for i in range(len(t_states)):
        state, policy, value = t_states[i], t_policies[i], t_values[i]
        aug_states, aug_policies = data_augmentation_single(state, policy, board_size, idx=idx)
        for j in range(len(aug_states)):
            new_states.append(aug_states[j])
            new_policies.append(aug_policies[j])
            new_values.append(value)
    new_states, new_policies, new_values = np.array(new_states), np.array(new_policies), np.array(new_values)
    return reshape_data_for_network(new_states, new_policies, new_values, board_size, input_planes)


#################################################
# Reshaping
#################################################

def reshape_data_for_network(states, policies, values, board_size, input_planes):
    t_states = np.reshape(states, (-1, board_size, board_size, input_planes))
    t_policies = np.reshape(policies, (-1, board_size ** 2 + 1))
    t_values = np.reshape(values, (-1, 1))
    return t_states, t_policies, t_values


def reshape_data_for_augmentation(states, policies, values, board_size, input_planes):
    t_states = np.reshape(states, (-1, 1, board_size, board_size, input_planes))
    t_policies = np.reshape(policies, (-1, 1, board_size ** 2 + 1))
    t_values = np.reshape(values, (-1,))
    return t_states, t_policies, t_values


def goban_1D_to_goban_2D(goban, size):
    return np.reshape(goban, (size, size))


def goban_to_nn_state(goban, board_size, dtype = bool):
    return np.reshape(goban, (1, -1, board_size, 1), dtype=dtype)


def goban_to_input_planes(goban, g_old, player_turn, size, dtype=bool):
    if player_turn == 1:
        goban = tuple(reversed(goban))
    goban = np.array(goban)
    g0 = goban_to_nn_state(goban[0], size)
    g1 = goban_to_nn_state(goban[1], size)
    g0_old = goban_to_nn_state(g_old[:, :, :, 0], size)
    g1_old = goban_to_nn_state(g_old[:, :, :, 1], size)
    goban = np.concatenate([g0, g0_old, g1, g1_old], axis=3).astype(dtype)
    g_old = np.concatenate([g1, g0], axis=3)
    return goban, g_old


def goban_split_to_planes(goban,turn, size, dtype=bool):
    """
    Tranforms a goban splitted into a feature to the NN
    :param goban: list of (list of bool, list of bool)
    :param turn: bool true if black turn, false if white turn
    :param size: the size of the goban
    :param dtype: the type for the numpy array
    :return: the ndarray formatted for the NN.
    """
    if turn == 1:
        goban = tuple(reversed(goban))
    goban = np.array(goban)
    g0 = goban_to_nn_state(goban[0], size, dtype)
    g1 = goban_to_nn_state(goban[1], size, dtype)
    goban = np.concatenate([g0, g1], axis=3)

    return


def add_player_feature_plane(goban, board_size, player_turn):
    player_feature_plane = np.full((board_size, board_size), player_turn)
    player_feature_plane = goban_to_nn_state(player_feature_plane, board_size)
    planes = np.concatenate([goban, player_feature_plane], axis=3)
    return planes


#################################################
# Other
#################################################

def letter_to_number(letter):
    return ord(letter) - 97


def move_scalar_to_tuple(move, board_size):
    return int(move / board_size), move % board_size
