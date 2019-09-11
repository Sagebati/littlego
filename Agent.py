import random

from libgoban import IGame


class Agent:
    def select_move(self, state):
        pass


class RandomAgent(Agent):
    def select_move(self, state: IGame):
        legals = state.legals()
        random.shuffle(legals)
        for p in legals:
            if not state.is_point_an_eye(p, state.turn()):
                return p  # if the point is not an eye the choose it
        else:
            return None  #
