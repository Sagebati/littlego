import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import l2_regularizer, xavier_initializer, fully_connected, flatten

#--------------------------------------------
#---------------- Activation ----------------
#--------------------------------------------

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

#various implementations of this exist, particularly in how the negatives are calculated
#for more, read https://arxiv.org/abs/1502.01852
def parametric_relu(X, regularizer=None, name="parametric_relu"):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alphas',
                                 regularizer=regularizer,
                                 dtype=X.dtype,
                                 shape=X.get_shape().as_list()[-1], 
                                 initializer=tf.constant_initializer(0.01))
        positives = tf.nn.relu(X)
        negatives = alphas*(tf.subtract(X, tf.abs(X))) * 0.5
        return positives + negatives

#this implementation assumes alpha will be less than 1. It would be stupid if it weren't.
def leaky_relu(X, alpha=0.2):
    return tf.maximum(X, alpha*X)



#------------------------------------------------
#---------------- Network Layers ----------------
#------------------------------------------------

#https://arxiv.org/abs/1302.4389
#This doesn't need a scope. There aren't trainable parameters here. It's just a pool
def maxout(X, num_maxout_units, axis=None):
    input_shape = X.get_shape().as_list()
    
    axis = -1 if axis is None else axis
    
    num_filters = input_shape[axis]
    
    if num_filters % num_maxout_units != 0:
        raise ValueError("num filters (%d) must be divisible by num maxout units (%d)" % (num_filters, num_maxout_units))
    
    output_shape = input_shape.copy()
    output_shape[axis] = num_maxout_units
    output_shape += [num_filters // num_maxout_units]
    return tf.reduce_max(tf.reshape(X, output_shape), -1, keep_dims=False)

def conv(X,
         output_filter_size,
         kernel=[5,5],
         strides=[2,2],
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
        
        return tf.nn.bias_add( tf.nn.conv2d(X,
                                            W,
                                            strides=[1,strides[0],strides[1],1],
                                            padding='SAME',
                                            name="conv2d"), b)



# ------------------------------
# W : input size (width or height)
# F : filters size
# P : padding
# S : stride
def conv_out_size(W, F, P, S):
	return (W - F + 2*P) / S + 1

def basic_layer(inputs, weights, biases, activation, use_batch_norm, drop_out, is_train):
	layer = tf.matmul(inputs, weights) + biases
	if use_batch_norm:
		layer = tf.layers.batch_normalization(layer, training=is_train)
	layer = activation(layer)			
	layer = tf.layers.dropout(layer, rate=drop_out, training=is_train)
	return layer

def conv_layer(inputs, filters, kernel, stride, activation, layer_name, use_batch_norm, drop_out, is_train):
	layer = conv(inputs,
				filters,
				kernel=[kernel, kernel],
				strides=[stride, stride],
				w_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
				name=layer_name)
	if use_batch_norm:
		layer = tf.layers.batch_normalization(layer, training=is_train)				
	layer = activation(layer)
	layer = tf.layers.dropout(layer, rate=drop_out, training=is_train)
	return layer

def residual_conv_block(inputs, filters, kernel, stride, activation, layer_name, use_batch_norm, drop_out, is_train):
	layer = conv_layer(inputs, filters, kernel, stride, activation, layer_name+"_1", use_batch_norm, drop_out, is_train)
	layer = conv_layer(layer, filters, kernel, stride, tf.identity, layer_name+"_2", use_batch_norm, drop_out, is_train)
	layer += inputs
	layer = activation(layer)
	return layer



#----------------------------------------------------------
#---------------- Noise and transformation ----------------
#----------------------------------------------------------

def dihedral_transformation(plane, k_rotate, reflection=False):
	new_plane = np.copy(plane)
	if reflection:
		new_plane = np.flip(new_plane, 1)
	new_plane = np.rot90(new_plane, k_rotate)
	return new_plane	
	
def dirichlet_noise(plane, alpha, epsilon):
	out = np.copy(plane)
	alphas = np.full(plane.shape, alpha)
	out = (1-epsilon) * out + epsilon * np.random.dirichlet(alphas)
	return out
