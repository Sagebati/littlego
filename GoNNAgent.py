import numpy as np
from libgoban import IGame

import ops
from Agent import Agent
from GoNeuralNetwork import GoNeuralNetwork


class GoNNAgent(Agent):

    def __init__(self, board_size):
        print("--- Initialization of Go Agent")
        self.board_size = board_size
        self.neural_network = GoNeuralNetwork(board_size)
        self.g_old = np.full((1, board_size, board_size, 2), 0)
        print("Initialized - Agent")

    def get_move(self, state: IGame):
        """if player_turn == 1:
            goban = tuple(reversed(goban))
        goban = np.array(goban)
        g0 = ops.goban_to_nn_state(goban[0], self.board_size)
        g1 = ops.goban_to_nn_state(goban[1], self.board_size)
        goban = np.concatenate([g0, g1], axis=3)"""
        goban = state.raw_goban_split()
        player_turn = state.turn()
        goban, self.g_old = ops.goban_to_input_planes(goban, self.g_old, player_turn, self.board_size, dtype=int)

        player_feature_plane = np.full([self.board_size, self.board_size], player_turn)
        player_feature_plane = ops.goban_to_nn_state(player_feature_plane, self.board_size)
        planes = np.concatenate([goban, player_feature_plane], axis=3)

        p, v = self.neural_network.get_move(planes, player_turn, state.legals())
        move = np.argmax(p)
        return move

    def end_game(self, winner):
        self.neural_network.save_in_replay_memory(winner)
