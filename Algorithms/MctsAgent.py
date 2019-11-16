import math
import random
from enum import Enum

from libgoban import IGame

from Agent import Agent, RandomAgent


class Player(Enum):
    Black = "B"
    White = "W"


def game_outcome_to_player(x):
    if x[0] == -1 or x[1] == -1:
        if x[0] == -1:
            return Player.Black
        else:
            return Player.White
    if x[0] > x[1]:
        return Player.Black
    else:
        return Player.White


def bool_to_player(b) -> Player:
    if b:
        return Player.White
    else:
        return Player.Black


# tag::mcts-node[]


class MCTSNode:
    def __init__(self, game_state: IGame, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = {
            Player.Black: 0,
            Player.White: 0,
        }
        self.num_rollouts = 0
        self.children = []
        self.unvisited_moves = game_state.legals()

    # end::mcts-node[]

    # tag::mcts-add-child[]
    def add_random_child(self):
        index = random.randint(0, len(self.unvisited_moves) - 1)
        new_move = self.unvisited_moves.pop(index)
        new_game_state = self.game_state.play_and_clone(new_move)
        new_node = MCTSNode(new_game_state, self, new_move)
        self.children.append(new_node)
        return new_node

    # end::mcts-add-child[]

    # tag::mcts-record-win[]
    def record_win(self, winner):
        self.win_counts[winner] += 1
        self.num_rollouts += 1

    # end::mcts-record-win[]

    # tag::mcts-readers[]
    def can_add_child(self):
        return len(self.unvisited_moves) > 0

    def is_terminal(self):
        return self.game_state.over()

    def winning_frac(self, player):
        return float(self.win_counts[player]) / float(self.num_rollouts)


# end::mcts-readers[]


class MCTSAgent(Agent):
    def __init__(self, num_rounds, temperature):
        self.num_rounds = num_rounds
        self.temperature = temperature

    # tag::mcts-signature[]
    def select_move(self, game_state: IGame):
        root = MCTSNode(game_state)
        # end::mcts-signature[]

        # tag::mcts-rounds[]
        for i in range(self.num_rounds):
            node = root
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)

            # Add a new child node into the tree.
            if node.can_add_child():
                node = node.add_random_child()

            # Simulate a random game from this node.
            winner = self.simulate_random_game(node.game_state)

            # Propagate scores back up the tree.
            while node is not None:
                node.record_win(winner)
                node = node.parent
        # end::mcts-rounds[]

        scored_moves = [
            (child.winning_frac(bool_to_player(game_state.turn())), child.move, child.num_rollouts)
            for child in root.children
        ]
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        for s, m, n in scored_moves[:10]:
            print('%s - %.3f (%d)' % (m, s, n))

        # tag::mcts-selection[]
        # Having performed as many MCTS rounds as we have time for, we
        # now pick a move.
        best_move = None
        best_pct = -1.0
        for child in root.children:
            child_pct = child.winning_frac(bool_to_player(game_state.turn()))
            if child_pct > best_pct:
                best_pct = child_pct
                best_move = child.move
        print('Select move %s with win pct %.3f' % (best_move, best_pct))
        return best_move

    # end::mcts-selection[]

    # tag::mcts-uct[]
    def select_child(self, node):
        """Select a child according to the upper confidence bound for
        trees (UCT) metric.
        """
        total_rollouts = sum(child.num_rollouts for child in node.children)
        log_rollouts = math.log(total_rollouts)

        best_score = -1
        best_child = None
        # Loop over each child.
        for child in node.children:
            # Calculate the UCT score.
            win_percentage = child.winning_frac(bool_to_player(node.game_state.turn))
            exploration_factor = math.sqrt(log_rollouts / child.num_rollouts)
            uct_score = win_percentage + self.temperature * exploration_factor
            # Check if this is the largest we've seen so far.
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child

    # end::mcts-uct[]

    @staticmethod
    def simulate_random_game(game):
        randombot = RandomAgent()
        while not game.over():
            bot_move = randombot.select_move(game)
            game.play(bot_move)
        return game_outcome_to_player(game.outcome())
