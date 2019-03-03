import random
from libshusaku import IGame
from GoNeuralNetwork import GoNeuralNetwork
import numpy as np

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



board_size = 13

goAgent = GoNNAgent(board_size)

turn_max = 666

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
