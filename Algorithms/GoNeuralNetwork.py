# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import ops
from SarstReplayMemory import SarstReplayMemory



#################################################
# Hyper-Parameters
#################################################

# --- Network parameters ---
activation = tf.nn.leaky_relu
# activation = tf.nn.selu
# activation = tf.nn.elu
# activation = tf.nn.relu

batch_size = 32             # total batch size
print(batch_size)
memory_capacity = 500000    # The size of the SarstReplayMemory class
useLSTM = False             # let False | TODO - implement LSTM
trace_length = 1

# weight_initializer = tf.truncated_normal_initializer()
# weight_initializer = tf.contrib.layers.xavier_initializer()
weight_initializer = tf.contrib.layers.variance_scaling_initializer()  # He_initializer
# weight_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN') # for selu

# - Optimizer
learning_rate = 0.001
decay_learning_rate = True
learning_rate_decay = 0.1
learning_rate_decay_steps = 100e3
learning_rate_min = 0.00001
momentum = 0.9

# - Regularization
l2_beta = 0.001
print(l2_beta)
useBatchNorm = True
drop_out = 0.                   # conv drop out
head_drop_out = drop_out        # dense drop out
use_gradient_clipping = False
clip_by_norm = True             # or by value
gradient_clipping_norm = 5.0

# - Conv "Tower" parameters
input_planes = 5
filters = 64
kernel_size = 3 # F
stride = 1      # S
num_blocks = 5  # each block has 2 conv layers

# - Policy head parameters
p_filters = 32
p_kernel_size = 1               # F
p_stride = 1                    # S
p_activation = tf.nn.softmax    # output policy activation

# - Value head parametersd
v_filters = 32
v_kernel_size = 1           # F
v_stride = 1                # S
v_activation = tf.nn.tanh   # output value activation
v_dense_size = 256

# --- MCTS parameters ---
c_puct = 4

# - Dirichlet noise
dirichlet_alpha = 0.03
dirichlet_epsilon = 0.25

# --- SaveFiles ---
modelCheckpoint = "./modelWeights.ckpt"
hyperparametersFile = "./modelParameters.gonn"
memoryFile = "./modelMemory.gonn"

# --- Frequencies ---
save_model_frequency = 1000
report_frequency = 1


# -----------------------------

class GoNeuralNetwork:

    def __init__(self, board_size):
        print("--- Initialization of Neural Network")

        # ----------------------------------------
        # Board size:
        self.board_size = board_size
        self.input_size = self.board_size * self.board_size
        self.input_shape = [self.board_size, self.board_size, input_planes]
        self.policy_size = self.input_size + 1

        self.network_inputs = {}
        self.memory_states = []
        self.memory_policies = []
        self.player_turns = []
        self.memory_loss = []
        self.total_iterations = 0
        self.temp_loss = 0
        self.total_games = 0
        # ----------------------------------------

        with tf.device('/gpu:0'):
            # with tf.Session() as session: # this line doesn't work when python is embedded
            session = tf.Session()
            self.init_network(session)

        self.saver = tf.train.Saver()  # tf.all_variables()
        self.restore_model()

    def init_network(self, tf_session):
        self.session = tf_session  # a tensorflow session

        self.replay_memory = SarstReplayMemory(memory_capacity,
                                               self.input_shape,
                                               [self.policy_size],
                                               useLSTM,
                                               trace_length,
                                               prioritize=True)
        print("Initialized - SARST Replay Memory")

        self.neural_network()
        print("Initialized - Neural Network")


    #################################################
    # Neural Network
    #################################################
    def neural_network(self):
        # Construct the two networks with identical architectures
        self.build_network('GoNeuralNetwork')

        # create the optimizer in the model
        self.build_optimizer()

        # initialize all these variables, mostly with xavier initializers
        init_op = tf.global_variables_initializer()

        # Ready to train
        self.session.run(init_op)

    # This function will be used to build neural network
    def build_network(self, scope_name):
        net_shape = [None] + [s for s in self.input_shape]
        print(net_shape)
        with tf.variable_scope(scope_name):
            self.is_train = tf.placeholder(tf.bool, name="is_train");
            self.global_step = tf.placeholder(tf.int32, name="global_step")

            # Input Layer
            self.network_inputs[scope_name] = tf.placeholder(tf.float32, shape=net_shape, name="inputs")

            # Conv Layers "Tower"
            conv = ops.conv_layer(self.network_inputs[scope_name], filters, kernel_size, stride, activation,
                                  "conv_first", useBatchNorm, drop_out, self.is_train, weight_initializer)
            for i in range(num_blocks):
                conv = ops.residual_conv_block(conv, filters, kernel_size, stride, activation, "conv" + str(i),
                                               useBatchNorm, drop_out, self.is_train, weight_initializer)

            # Policy and value heads
            # - Compute conv output size
            tower_conv_out_size = ops.conv_out_size(self.input_size, kernel_size, 1, stride)
            # TODO - manage correctly padding (if stride and/or filter size change)
            value_conv_out_size = ops.conv_out_size(tower_conv_out_size, v_kernel_size, 0, v_stride) * v_filters
            policy_conv_out_size = ops.conv_out_size(tower_conv_out_size, p_kernel_size, 0, p_stride) * p_filters

            # - Declare weights and biases
            weights = {
                'policy': tf.get_variable('w_policy', shape=[policy_conv_out_size, self.policy_size],
                                          initializer=tf.contrib.layers.xavier_initializer()),
                'value': tf.get_variable('w_value', shape=[value_conv_out_size, v_dense_size],
                                         initializer=weight_initializer),
                'value_out': tf.get_variable('w_value_out', shape=[v_dense_size, 1],
                                             initializer=tf.contrib.layers.xavier_initializer()
                                             # tf.truncated_normal_initializer()
                                             )
            }
            biases = {
                'policy': tf.get_variable('b_policy', shape=[self.policy_size],
                                          initializer=tf.constant_initializer(0.0)),
                'value': tf.get_variable('b_value', shape=[v_dense_size],
                                         initializer=tf.constant_initializer(0.0)),
                'value_out': tf.get_variable('b_value_out', shape=[1],
                                             initializer=tf.constant_initializer(0.0))
            }

            # - Policy head
            policy_conv = ops.conv_layer(conv, p_filters, p_kernel_size, p_stride, activation, "policy_conv",
                                         useBatchNorm, drop_out, self.is_train, weight_initializer)
            policy_conv = tf.contrib.layers.flatten(policy_conv)
            self.policy_out = ops.basic_layer(policy_conv, weights["policy"], biases["policy"], tf.identity, False, 0.0,
                                              self.is_train)
            self.policy_out_prob = p_activation(self.policy_out)

            # - Value head
            value_conv = ops.conv_layer(conv, v_filters, v_kernel_size, v_stride, activation, "value_conv",
                                        useBatchNorm, drop_out, self.is_train, weight_initializer)
            value_conv = tf.contrib.layers.flatten(value_conv)
            value_out = ops.basic_layer(value_conv, weights["value"], biases["value"], activation, False, head_drop_out,
                                        self.is_train)
            self.value_out = ops.basic_layer(value_out, weights["value_out"], biases["value_out"], v_activation, False,
                                             0.0, self.is_train)

    # ----- Optimizer -----
    def learning_rate_scheduling(self):
        if decay_learning_rate:
            lr = tf.maximum(learning_rate_min, 
							tf.train.exponential_decay(learning_rate,
                                                    self.global_step,
                                                    learning_rate_decay_steps,
                                                    learning_rate_decay))
        else:
            lr = tf.add(learning_rate, 0) # to make it a tf tensor
        return lr

    def build_optimizer(self):
        with tf.variable_scope('optimizer'):
            # Define placeholder
            # self.target_v = tf.placeholder(tf.float32, shape=[None], name="target_v")
            self.target_v = tf.placeholder(tf.float32, shape=[None, 1], name="target_v")
            self.target_p = tf.placeholder(tf.float32, shape=[None, self.policy_size], name="target_p")

            # Loss
            loss_v = tf.reduce_mean(tf.square(tf.subtract(self.target_v, self.value_out)))
            loss_p = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target_p, logits=self.policy_out))
            self.loss_op = tf.add(loss_v, loss_p)
            
            # L2 regularization loss
            if l2_beta != 0.:
                l2 = l2_beta * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                         if not ("b_" in v.name)])
                # self.loss_op = tf.add(self.loss_op, tf.reduce_mean(l2))
                self.loss_op = tf.add(self.loss_op, l2)

            # Learning rate scheduling
            self.learning_rate = self.learning_rate_scheduling()

            # Optimizer
            # opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            # opt = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum)
            # opt = tf.train.MomentumOptimizer(learning_rate, momentum)

            # Gradient clipping and optimization
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):  # Needed for batch normalization
                if use_gradient_clipping:
                    gvs = opt.compute_gradients(self.loss_op)
                    if clip_by_norm:
                        grad, vs = zip(*gvs)
                        grad, _ = tf.clip_by_global_norm(grad, gradient_clipping_norm)
                        capped_gvs = zip(grad, vs)
                    else:
                        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
                    self.optimizer = opt.apply_gradients(capped_gvs)
                else:
                    self.optimizer = opt.minimize(self.loss_op)

            # Accuracies
            policy_correct_prediction = tf.equal(tf.argmax(self.policy_out_prob, 1), tf.argmax(self.target_p, 1))
            self.policy_accuracy = tf.reduce_mean(tf.cast(policy_correct_prediction, "float"))
            self.value_accuracy = 1. - tf.reduce_mean(tf.abs(tf.subtract(self.target_v, self.value_out))) / 2.
            self.value_error = loss_v

    # ----- Minibatch -----
    def run_minibatch(self):
        if self.replay_memory.memory_size >= batch_size * trace_length:
            state, target_p, target_v = self.replay_memory.get_batch_sample(batch_size)
            loss, _, _ = self.train(state, target_p, target_v, self.total_iterations)
            if self.total_iterations % report_frequency == 0:
                print("\nMinibatch {} : \nloss = {}".format(self.total_iterations, loss))
                print("memory = {}".format(self.replay_memory.memory_size))
                print()
            self.total_iterations += 1
            self.temp_loss += loss
            if self.total_iterations % save_model_frequency == 0:
                # self.memory_loss.append(self.temp_loss / save_model_frequency)
                np.append(self.memory_loss, self.temp_loss / save_model_frequency)
                self.temp_loss = 0
                self.save_model()

    def train(self, state, target_p, target_v, global_step):
        _, loss, p_acc, v_err = self.session.run(
            [self.optimizer, self.loss_op, self.policy_accuracy, self.value_error],
            {self.network_inputs['GoNeuralNetwork']: state,
             self.target_v: target_v,  
             self.target_p: target_p,
             self.is_train: True,
             self.global_step: global_step
             })
        return loss, p_acc, v_err

    # ----- Feed Forward (without training) -----
    def feed_forward(self, state):
        p, v = self.session.run(
            [self.policy_out_prob, self.value_out],
            {self.network_inputs['GoNeuralNetwork']: state,
             self.is_train: False,
             self.global_step: self.total_iterations
             })
        return p, v

    def feed_forward_value(self, state):
        v = self.session.run(
            [self.value_out],
            {self.network_inputs['GoNeuralNetwork']: state,
             self.is_train: False,
             self.global_step: self.total_iterations
             })
        return v

    def feed_forward_accuracies(self, state, target_p, target_v, global_step):
        p_acc, v_err, p_out, v_out = self.session.run(
            [self.policy_accuracy, self.value_error, self.policy_out_prob, self.value_out],
            {self.network_inputs['GoNeuralNetwork']: state,
             self.target_v: target_v, 
             self.target_p: target_p,
             self.is_train: False,
             self.global_step: global_step
             })
        return p_acc, v_err, p_out, v_out

    # ----- Policy Improvement Operators -----
    def remove_illegal(self, legals, p):
        new_p = np.full(p.shape, -1.)
        for move in legals:
            s_move = move[0] * self.board_size + move[1]
            new_p[0][s_move] = p[0][s_move]
        return new_p

    def weak_mcts(self, planes, player_turn, legals, p):
        planes = np.copy(planes)
        new_p = np.full(p.shape, 0.)
        num_boards = (input_planes - 1) / 2

        # Reverse player feature plane
        p_plane = input_planes - 1
        planes[:, :, :, p_plane] = np.full(planes[:, :, :, p_plane].shape, (player_turn + 1) % 2)

        # Swap board planes
        tmp = np.copy(planes[:, :, :, 0])
        planes[:, :, :, 0] = planes[:, :, :, 1]
        planes[:, :, :, 1] = tmp

        # Simulates the legal move and get value
        for move in legals:
            # planes[0][move[0]][move[1]][0] = player_turn+1
            last_plane = int(input_planes - 1 - num_boards)
            planes[0][move[0]][move[1]][last_plane] = 1
            t_v = self.feed_forward_value(planes)
            planes[0][move[0]][move[1]][last_plane] = 0
            s_move = move[0] * self.board_size + move[1]
            t_v = t_v[0][0][0]
            new_p[0][s_move] = (t_v * (-1.) + 2.)  # + p[0][s_move]

        # Simulates pass move and get value
        t_v = self.feed_forward_value(planes)
        t_v = t_v[0][0][0]
        new_p[0][self.input_size] = (t_v * (-1.) + 2.)  # + p[0][self.input_size]

        # Dirichlet noise
        new_p[0] = ops.dirichlet_noise(new_p[0], dirichlet_alpha, dirichlet_epsilon)

        # Activate
        new_p = ops.softmax(new_p)

        return new_p

    def get_move(self, planes, player_turn, legals):
        self.run_minibatch()
        # TODO - Dihedral Transformation on planes (and reverse transformation for p)
        p, v = self.feed_forward(planes)

        # Policy Improvement Operator
        # TODO - (real) MCTS policy improvement
        # p = self.remove_illegal(legals, p)
        p = self.weak_mcts(planes, player_turn, legals, p)

        # Data saving and augmentation
        self.save_in_self_memory(planes, p, player_turn)

        return p, v


    #################################################
    # Save & Restore 
    #################################################
    def save_one_in_self_memory(self, state, policy, player_turn):
        self.memory_states.append(state)
        self.memory_policies.append(policy)
        self.player_turns.append(player_turn)

    def save_in_self_memory(self, planes, policy, player_turn):
        # self.save_one_in_self_memory(planes, p, player_turn)

        # Data augmentation
        out_planes, out_policies = ops.data_augmentation_single(planes, policy, self.board_size)
        for i in range(len(out_planes)):
            new_planes = out_planes[i]
            new_policy = out_policies[i]
            # Save to self memory
            self.save_one_in_self_memory(new_planes, new_policy, player_turn)

    def save_in_replay_memory(self, winner):
        for i in range(len(self.memory_states)):
            state = self.memory_states[i]
            policy = self.memory_policies[i]
            player_turn = self.player_turns[i]
            value = 0 if winner == 2 else 1 if player_turn == winner else -1
            self.replay_memory.add_to_memory(state, policy, value, self.total_iterations)
        self.memory_states = []
        self.memory_policies = []
        self.player_turns = []
        self.total_games += 1

    def save_model(self, save_memory=True):
        # TODO - save with model name
        # Save model
        self.saver.save(self.session, modelCheckpoint)
        print("=== Model saved as \"{}\" ===".format(modelCheckpoint))
        # Save memory
        if save_memory:
            self.save_memory(memoryFile)
        # Save parameters
        np.savez(hyperparametersFile,
                 total_iterations=self.total_iterations,
                 memory_loss=self.memory_loss,
                 total_games=self.total_games)
        print("=== Parameters saved as \"{}.npz\" ===\n".format(hyperparametersFile))

    # TODO - save rnn_state_train

    def save_memory(self, memoryFile):
        self.replay_memory.save_memory(memoryFile)

    def restore_model(self):
        if (tf.train.checkpoint_exists("checkpoint")):
            # Restore model
            self.saver.restore(self.session, modelCheckpoint)
            print("\n=== Model restored from \"{}\" ===".format(modelCheckpoint))
            # Restore memory
            self.restore_memory(memoryFile)
            # Restore parameters
            npzFile = np.load("{}.npz".format(hyperparametersFile))
            self.total_iterations = npzFile['total_iterations']
            self.memory_loss = npzFile['memory_loss']
            self.total_games = npzFile['total_games']
            print("=== Parameters restored from \"{}.npz\" ===\n".format(hyperparametersFile))
        """else:
            self.copy_prediction_parameters_to_target_network()"""

    def restore_memory(self, memoryFile):
        self.replay_memory.restore_memory(memoryFile)
        
    
    #################################################
    # Getter and Setter
    #################################################
    def get_learning_rate(self, global_step):
        lr = self.session.run([self.learning_rate], {self.global_step: global_step})
        return lr
