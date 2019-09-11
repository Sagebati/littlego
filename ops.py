import os

import numpy as np
import tensorflow as tf
from libgoban import IGame
from tensorflow.contrib.layers import xavier_initializer


# --------------------------------------------
# ---------------- Activation ----------------
# --------------------------------------------

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


# ------------------------------------------------
# ---------------- Network Layers ----------------
# ------------------------------------------------

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


# ------------------------------
# W : input size (width or height)
# F : filters size
# P : padding
# S : stride
def conv_out_size(W, F, P, S):
    return (W - F + 2 * P) / S + 1


def basic_layer(inputs, weights, biases, activation, use_batch_norm, drop_out, is_train):
    layer = tf.matmul(inputs, weights) + biases
    if use_batch_norm:
        layer = tf.layers.batch_normalization(layer, training=is_train)
    layer = activation(layer)
    layer = tf.layers.dropout(layer, rate=drop_out, training=is_train)
    return layer


def conv_layer(inputs, filters, kernel, stride, activation, layer_name, use_batch_norm, drop_out, is_train,
               weight_initializer=xavier_initializer()):
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


def residual_conv_block(inputs, filters, kernel, stride, activation, layer_name, use_batch_norm, drop_out, is_train,
                        weight_initializer=xavier_initializer()):
    layer = conv_layer(inputs, filters, kernel, stride, activation, layer_name + "_1", use_batch_norm, drop_out,
                       is_train, weight_initializer)
    layer = conv_layer(layer, filters, kernel, stride, tf.identity, layer_name + "_2", use_batch_norm, drop_out,
                       is_train, weight_initializer)
    layer += inputs
    layer = activation(layer)
    return layer


# ----------------------------------------------------------
# ---------------- Noise and transformation ----------------
# ----------------------------------------------------------

def dihedral_transformation(plane, k_rotate, reflection=False):
    new_plane = np.copy(plane)
    if reflection:
        new_plane = np.flip(new_plane, 1)
    new_plane = np.rot90(new_plane, k_rotate)
    return new_plane


def dirichlet_noise(plane, alpha, epsilon):
    out = np.copy(plane)
    alphas = np.full(plane.shape, alpha)
    out = (1 - epsilon) * out + epsilon * np.random.dirichlet(alphas)
    return out


# Data augmentation from raw neural network inputs
def data_augmentation(planes, policy, board_size):
    out_planes = []
    out_policies = []

    num_board_planes = planes.shape[3] - 1

    planes = np.copy(planes)
    p = np.copy(policy)
    p_pass = p[0][-1]
    t_p = np.reshape(p[0][:-1], (board_size, board_size))
    t_plane = {}
    for i in range(num_board_planes):
        t_plane[i] = np.reshape(np.copy(planes[:, :, :, i]), (board_size, board_size))
    for reflect in (False, True):
        for k_rotate in range(0, 4):
            # Rotate/reflect policy out
            new_p = dihedral_transformation(t_p, k_rotate, reflect)
            new_p = np.append(new_p, p_pass)
            new_p = np.reshape(new_p, (1, board_size * board_size + 1))

            # Rotate/reflect planes
            for i in range(num_board_planes):
                new_plane = dihedral_transformation(t_plane[i], k_rotate, reflect)
                new_plane = np.reshape(new_plane, (1, board_size, board_size))
                planes[:, :, :, i] = new_plane

            out_planes.append(np.copy(planes))
            out_policies.append(np.copy(new_p))

    return out_planes, out_policies


def letter_to_number(letter):
    return ord(letter) - 97


def goban_1D_to_goban_2D(goban, size):
    return np.reshape(goban, (size, size))


def goban_to_nn_state(goban, board_size):
    return np.reshape(goban, (1, -1, board_size, 1))


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


def move_scalar_to_tuple(move, board_size):
    return int(move / board_size), move % board_size


# ------------------------------------------
# ---------------- SGF File ----------------
# ------------------------------------------

def SGF_file_parser(file_name):
    with open(file_name, "r") as fichier:
        content = fichier.read()
        content = content.replace("[]", '\n  \n').replace('[', '\n').replace(']', '\n').replace(';', '\n')
        content = content.split("\n")
        content = list(filter(lambda a: a != '' and a != ')' and a != '(', content))
    return content


def SGF_file_to_dataset(file_name):
    content = SGF_file_parser(file_name)

    states = []
    policies = []
    values = []
    player_turn = []

    size = 19
    handicap = 0
    winner = 2
    points_or_resign = ""

    g = IGame(size)

    g_old = np.full((1, size, size, 2), 0)
    for i in range(len(content)):
        elem = content[i]
        # Board size
        if elem == "SZ":
            size = int(content[i + 1])
            g = IGame(size)
        # Handicap
        elif elem == "HA":
            handicap = int(content[i + 1])
        elif elem == "KM":
            komi = float(content[i + 1])
            g.set_komi(komi)
        # Result
        elif elem == "RE":
            splited = content[i + 1].split("+")
            winner, points_or_resign = splited[0], splited[1]
            winner = 0 if winner == "B" else 1 if winner == "W" else 2
        # Handicap moves
        elif elem == "AW" or elem == "AB":
            for h in range(handicap):
                x = letter_to_number(content[i + 1 + h][0])
                y = letter_to_number(content[i + 1 + h][1])
                g.play((x, y))
                g.play(None)
            g.play(None)  # Necessary because it's up to white to play
        # g.display_goban()
        # Moves
        elif elem == "W" or elem == "B":
            player = 0 if elem == "B" else 1
            # Make state
            # goban = goban_1D_to_goban_2D(g.goban(), size)
            goban = g.raw_goban_split()
            goban, g_old = goban_to_input_planes(goban, g_old, player, size)

            # Make policy
            policy = np.zeros(size * size + 1)
            if content[i + 1] == '  ':
                move = size * size
            else:
                x = letter_to_number(content[i + 1][0])
                y = letter_to_number(content[i + 1][1])
                move = x * size + y
            policy[move] = 1
            # Make value
            value = 0 if winner == 2 else 1 if winner == player else -1

            # Save data
            states.append(goban)
            policies.append(policy)
            values.append(value)
            player_turn.append(player)

            # Play move
            if move == size * size:
                g.play(None)
            else:
                g.play((x, y))
        # g.display_goban()

    if points_or_resign == "Resign" or points_or_resign == "R":
        g.resign(True if winner == 1 else False)
    print(file_name)
    print(str(winner) + " " + points_or_resign)
    print(g.outcome())

    return states, policies, values, player_turn


def SGF_folder_to_dataset(folder_name, out):
    all_states = []
    all_policies = []
    all_values = []
    all_turn = []

    i = 0
    for file_name in os.listdir(folder_name):
        if file_name[-4:] == ".sgf":
            print(i)
            i += 1
            file_name = folder_name + file_name
            states, policies, values, player_turn = SGF_file_to_dataset(file_name)
            for state in states:
                all_states.append(state)
            for policy in policies:
                all_policies.append(policy)
            for value in values:
                all_values.append(value)
            for turn in player_turn:
                all_turn.append(turn)

    np.savez(out + "dataset",
             states=all_states,
             policies=all_policies,
             values=all_values,
             player_turn=all_turn)


def SGF_folder_rule_filter(folder_name, rule_filter):
    for file_name in os.listdir(folder_name):
        if file_name[-4:] == ".sgf":
            is_filter = True
            file_name = folder_name + file_name
            content = SGF_file_parser(file_name)
            for i in range(len(content)):
                elem = content[i]
                if elem == "RU" and content[i + 1] == rule_filter:
                    is_filter = False
            if is_filter:
                print("remove {}".format(file_name))
                os.remove(file_name)
