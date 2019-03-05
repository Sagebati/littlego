import random
import sys
from libshusaku import IGame
from GoNeuralNetwork import GoNeuralNetwork
import numpy as np
import ops
from random import randint, shuffle

class GoNNAgent():

	def __init__(self, board_size):
		print("--- Initialization of Go Agent")
		self.board_size = board_size
		self.neural_network = GoNeuralNetwork(board_size)
		print("Initialized - Agent")

	def goban_to_nn_state(self, goban):
		goban = np.reshape(goban, (1, -1, self.board_size, 1))
		return goban

	def get_move(self, goban, player_turn, legals):
		goban = self.goban_to_nn_state(goban)
		player_feature_plane = np.full([self.board_size, self.board_size], player_turn)
		player_feature_plane = self.goban_to_nn_state(player_feature_plane)
		planes = np.concatenate([goban, player_feature_plane], axis=3)
		
		p, v = self.neural_network.get_move(planes, player_turn, legals)
		move = np.argmax(p)
		return move
	
	def move_scalar_to_tuple(self, move):
		t_move = (int(move / self.board_size) , move % self.board_size)
		return t_move
		
	def end_game(self, winner):
		self.neural_network.save_in_replay_memory(winner)
		
	def supervised_training(self, dataset, k_fold = 0):
		# Training parameters
		epoch = 10000
		report_frequency = 5
		batch_size = 32
		test_ratio = 1/30 # DeepMind paper
		k = k_fold # k-fold cross validation
		data_size = 2000

		maxK = int(1/test_ratio)
		k = max(k, maxK-1)
		
		# Load dataset
		print("Data loading")
		npzfile = np.load(dataset)
		t_states = npzfile['states']
		t_policies = npzfile['policies']
		t_values = npzfile['values']
		player_turn = npzfile['player_turn']
		
		# TODO - remove the four next lines to work on the full dataset
		t_states = t_states[:data_size]
		t_policies = t_policies[:data_size]
		t_values = t_values[:data_size]
		player_turn = player_turn[:data_size]

		# Shape states to neural network input shape
		print("Data shaping")
		tt_states = []
		tt_policies = []
		for i in range(len(t_states)):
			player_feature_plane = np.full(t_states[i].shape, player_turn[i])
			t_state = self.goban_to_nn_state(t_states[i])
			player_feature_plane = self.goban_to_nn_state(player_feature_plane)
			tt_states.append(np.concatenate([t_state, player_feature_plane], axis=3))
			tt_policies.append(np.reshape(t_policies[i], (1, self.board_size ** 2 + 1)))
		t_states = np.array(tt_states)
		t_policies = np.array(tt_policies)
		
		# Data augmentation 
		print("Data augmentation")
		states, policies, values = [], [], []
		for i in range(len(t_states)):
			state, policy, value = t_states[i], t_policies[i], t_values[i]
			new_states, new_policies = ops.data_augmentation(state, policy, self.board_size)
			for j in range(len(new_states)):
				states.append(new_states[j])
				policies.append(new_policies[j])
				values.append(value)

		# Shuffle
		print("Data shuffling")
		temp = list(zip(states, policies, values))
		shuffle(temp)
		states, policies, values = zip(*temp)
		states, policies, values = np.array(states), np.array(policies), np.array(values)	
		
		# Data splitting
		print("Data splitting")
		len_dataset = len(values)		
		test_size = int(len_dataset * test_ratio)		
		if test_size != 0:
			b_split = int(len_dataset * test_ratio * k)
			e_split = int((b_split + len_dataset * test_ratio) % len_dataset)
			test_states, test_policies, test_values = states[b_split:e_split], policies[b_split:e_split], values[b_split:e_split]
			train_states = np.concatenate([states[0:b_split], states[e_split:]])
			train_policies = np.concatenate([policies[0:b_split], policies[e_split:]])
			train_values = np.concatenate([values[0:b_split], values[e_split:]])
			
			test_states = np.reshape(test_states, (-1, self.board_size, self.board_size, 2))
			test_policies = np.reshape(test_policies, (-1, self.board_size ** 2 + 1))			
			
			validation_states, validation_policies, validation_values = test_states, test_policies, test_values

		# Training
		print("Training")
		len_train = train_states.shape[0]
		for i in range(epoch):
			# Get batch
			if(batch_size == len_train):
				batch_states, batch_policies, batch_values = train_states, train_policies, train_values
			else:
				idx = []
				while len(idx) < batch_size:
					idx.append(np.random.randint(low=0, high=len_train))
				batch_states, batch_policies, batch_values = train_states[idx], train_policies[idx], train_values[idx]
			batch_states = np.reshape(batch_states, (-1, self.board_size, self.board_size, 2))
			batch_policies = np.reshape(batch_policies, (-1, self.board_size ** 2 + 1))

			# Train model on this batch
			loss, p_acc, v_acc = self.neural_network.train(batch_states, batch_policies, batch_values, epoch)
			# Validation
			if test_size != 0:
				val_p_acc, val_v_acc = self.neural_network.feed_forward_accuracies(validation_states, validation_policies, validation_values, epoch)
			# Print results
			if i % report_frequency == 0:
				print("\nMinibatch {} : \nloss = {}".format(i, loss))
				print("TRAINING  :\npolicy accuracy = {:.4f}\nvalue  accuracy = {:.4f}".format(p_acc, v_acc))
				if test_size != 0:
					print("VALIDATION:\npolicy accuracy = {:.4f}\nvalue  accuracy = {:.4f}".format(val_p_acc, val_v_acc))	
				print()		

		print("Optimization Finished!")
		test_p_acc, test_v_acc = self.neural_network.feed_forward_accuracies(test_states, test_policies, test_values, 0)
		print("TEST      :\npolicy accuracy = {:.4f}\nvalue  accuracy = {:.4f}".format(test_p_acc, test_v_acc))



if __name__ == '__main__':
	
	if len(sys.argv) != 1:
	# Supervised training
		board_size = 19
		goAgent = GoNNAgent(board_size)	
		goAgent.supervised_training(sys.argv[1])
	else:
	# Reinforcement training
		board_size = 13
		goAgent = GoNNAgent(board_size)

		turn_max = int(board_size ** 2 * 2.5)

		for i in range(1000):

			g = IGame(board_size)
			g.display()

			total_turn = 0
			while not g.over():
				print("game {} - total_turn = {}".format(i, total_turn))
				goban = g.goban()
				move = goAgent.get_move(goban, total_turn % 2, g.legals())
				t_move = goAgent.move_scalar_to_tuple(move)
				if t_move in g.legals():
					print(t_move)			
					g.play(t_move)
				else:
					print("(skip_move)")
					g.skip()
				g.display()
				total_turn += 1	
				if total_turn > turn_max:
					break

			if total_turn > turn_max:	
				winner = 2 # draw
			else:
				score = g.outcome()
				print(score)
				winner = 2 if score[0] == score[1] else 0 if score[0] > score[1] else 1			
			goAgent.end_game(winner)


"""l = g.legals()
print(l)
g.play(random.choice(l))
goban = g.goban()

print(goban)
print(p)"""

"""for i in range(0, 500):
    l = g.legals()
    g.play(random.choice(l))
    #print(g.goban())
    g.display()
g.display()"""
