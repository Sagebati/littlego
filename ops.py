import sys
import os
from libshusaku import IGame
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
	
def data_augmentation(planes, policy, board_size):
	out_planes = []
	out_policies = []
	
	planes = np.copy(planes)
	p = np.copy(policy)
	p_pass = p[0][-1]
	t_p = np.reshape(p[0][:-1], (board_size, board_size))
	t_plane = np.reshape(np.copy(planes[:,:,:,0]), (board_size, board_size))
	for reflect in (False, True):
		for k_rotate in range(0,4):
			# Rotate/reflect policy out
			new_p = dihedral_transformation(t_p, k_rotate, reflect)
			new_p = np.append(new_p, p_pass)
			new_p = np.reshape(new_p, (1, board_size*board_size+1))
			
			# Rotate/reflect planes
			new_plane = dihedral_transformation(t_plane, k_rotate, reflect)
			new_plane = np.reshape(new_plane, (1, board_size, board_size))
			planes[:,:,:,0] = new_plane
			
			out_planes.append(np.copy(planes))
			out_policies.append(np.copy(new_p))
				
	return out_planes, out_policies

def letter_to_number(letter):
	return ord(letter) - 97

def goban_1D_to_goban_2D(goban, size):
	return np.reshape(goban, (size, size))

#------------------------------------------
#---------------- SGF File ----------------
#------------------------------------------

def SGF_file_parser(file_name):
	fichier = open(file_name)
	content = fichier.read()
	fichier.close()
	content = content.replace("[]", '\n  \n').replace('[', '\n').replace(']', '\n').replace(';', '\n')
	content = content.split("\n")
	content = list(filter(lambda a: a != '' and a != ')' and a != '(', content))
	return content

def SGF_file_to_dataset(file_name):
	content = SGF_file_parser(file_name)
	
	states = []
	policies = []
	values = [] 
	
	size = 19
	handicap = 0
	winner = 2
	player_turn = 0
	
	g = IGame(size)
	
	for i in range(len(content)):
		elem = content[i]
		# Board size
		if elem == "SZ":
			size = int(content[i+1])
			g = IGame(size)
		# Handicap
		elif elem == "HA":
			handicap = int(content[i+1])
		# Result
		elif elem == "RE":
			winner = content[i+1].split("+")[0]
			winner = 0 if winner == "B" else 1 if winner == "W" else 2
		# Handicap moves
		elif elem == "AW" or elem == "AB":
			for h in range(handicap):
				x = letter_to_number(content[i+1+h][0])
				y = letter_to_number(content[i+1+h][1])
				g.play((x, y))
				g.skip()
			g.skip()
			g.display()
		# Moves
		elif elem == "W" or elem == "B":
			# Make state
			goban = goban_1D_to_goban_2D(g.goban(), size)		
			# Make policy
			policy = np.zeros(size*size+1)
			if content[i+1] == '  ':
				move = size*size
			else:
				x = letter_to_number(content[i+1][0])
				y = letter_to_number(content[i+1][1])			
				move = x * size + y
			policy[move] = 1
			# Make value
			player = 0 if elem == "B" else 1
			value = 0 if winner == 2 else 1 if winner == player else -1
			
			# Save data
			states.append(goban)
			policies.append(policy)
			values.append(value)
			
			# Play move
			if move == size*size:
				g.skip()
			else:
				g.play((x, y))
			g.display()
	
	print(file_name)
	print(winner)
	print(g.outcome())
	input()
	
	return states, policies, values	

def SGF_folder_to_dataset(folder_name):
	all_states = []
	all_policies= []
	all_values = []
	
	for file_name in os.listdir(folder_name):
		if file_name[-4:] == ".sgf":		
			file_name = folder_name+file_name
			states, policies, values = SGF_file_to_dataset(file_name)
			for state in states:
				all_states.append(state)
			for policy in policies:
				all_policies.append(policy)
			for value in values:
				all_values.append(value)

	np.savez(folder_name, 
			states = all_states,
			policies = all_policies,
			values = all_values)
			
def SGF_folder_rule_filter(folder_name, rule_filter):
	for file_name in os.listdir(folder_name):
		if file_name[-4:] == ".sgf":
			is_filter = True
			file_name = folder_name+file_name			
			#print("process {}".format(file_name))
			content = SGF_file_parser(file_name)
			for i in range(len(content)):
				elem = content[i]
				if elem == "RU" and content[i+1] == rule_filter:
					is_filter = False
			if is_filter:
				print("remove {}".format(file_name))
				os.remove(file_name)

if __name__ == '__main__':
	#SGF_folder_rule_filter(sys.argv[1], "Chinese")
	#SGF_folder_to_dataset(sys.argv[1])
	SGF_file_to_dataset(sys.argv[1])
	pass
