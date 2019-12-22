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


class AlphaGoNode:
    def __init__(self, parent=None, probability=1.0):
        self.parent = parent  # <1>
        self.children = {}  # <1>

        self.visit_count = 0
        self.q_value = 0
        self.prior_value = probability  # <2>
        self.u_value = probability  # <3>
# <1> Tree nodes have one parent and potentially many children.
# <2> A node is initialized with a prior probability.
# <3> The utility function will be updated during search.
# end::init_alphago_node[]

# tag::select_node[]
    def select_child(self):
        return max(self.children.items(),
                   key=lambda child: child[1].q_value + \
                   child[1].u_value)
# end::select_node[]

# tag::expand_children[]
    def expand_children(self, moves, probabilities):
        for move, prob in zip(moves, probabilities):
            if move not in self.children:
                self.children[move] = AlphaGoNode(probability=prob)
# end::expand_children[]

# tag::update_values[]
    def update_values(self, leaf_value):
        if self.parent is not None:
            self.parent.update_values(leaf_value)  # <1>

        self.visit_count += 1  # <2>

        self.q_value += leaf_value / self.visit_count  # <3>

        if self.parent is not None:
            c_u = 5
            self.u_value = c_u * np.sqrt(self.parent.visit_count) \
                * self.prior_value / (1 + self.visit_count)  # <4>

# <1> We update parents first to ensure we traverse the tree top to bottom.
# <2> Increment the visit count for this node.
# <3> Add the specified leaf value to the Q-value, normalized by visit count.
# <4> Update utility with current visit counts.
# end::update_values[]


# tag::alphago_mcts_init[]
class AlphaGoMCTS(Agent):
    def __init__(self, policy_agent, fast_policy_agent, value_agent,
                 lambda_value=0.5, num_simulations=1000,
                 depth=50, rollout_limit=100):
        self.policy = policy_agent
        self.rollout_policy = fast_policy_agent
        self.value = value_agent

        self.lambda_value = lambda_value
        self.num_simulations = num_simulations
        self.depth = depth
        self.rollout_limit = rollout_limit
        self.root = AlphaGoNode()
# end::alphago_mcts_init[]

# tag::alphago_mcts_rollout[]
    def select_move(self, game_state):
        for simulation in range(self.num_simulations):  # <1>
            current_state = game_state
            node = self.root
            for depth in range(self.depth):  # <2>
                if not node.children:  # <3>
                    if current_state.is_over():
                        break
                    moves, probabilities = self.policy_probabilities(current_state)  # <4>
                    node.expand_children(moves, probabilities)  # <4>

                move, node = node.select_child()  # <5>
                current_state = current_state.apply_move(move)  # <5>

            value = self.value.predict(current_state)  # <6>
            rollout = self.policy_rollout(current_state)  # <6>

            weighted_value = (1 - self.lambda_value) * value + \
                self.lambda_value * rollout  # <7>

            node.update_values(weighted_value)  # <8>
# <1> From current state play out a number of simulations
# <2> Play moves until the specified depth is reached.
# <3> If the current node doesn't have any children...
# <4> ... expand them with probabilities from the strong policy.
# <5> If there are children, we can select one and play the corresponding move.
# <6> Compute output of value network and a rollout by the fast policy.
# <7> Determine the combined value function.
# <8> Update values for this node in the backup phase
# end::alphago_mcts_rollout[]

# tag::alphago_mcts_selection[]
        move = max(self.root.children, key=lambda move:  # <1>
                   self.root.children.get(move).visit_count)  # <1>

        self.root = AlphaGoNode()
        if move in self.root.children:  # <2>
            self.root = self.root.children[move]
            self.root.parent = None

        return move
# <1> Pick most visited child of the root as next move.
# <2> If the picked move is a child, set new root to this child node.
# end::alphago_mcts_selection[]

# tag::alphago_policy_probs[]
    def policy_probabilities(self, game_state):
        encoder = self.policy._encoder
        outputs = self.policy.predict(game_state)
        legal_moves = game_state.legal_moves()
        if not legal_moves:
            return [], []
        encoded_points = [encoder.encode_point(move.point) for move in legal_moves if move.point]
        legal_outputs = outputs[encoded_points]
        normalized_outputs = legal_outputs / np.sum(legal_outputs)
        return legal_moves, normalized_outputs
# end::alphago_policy_probs[]

# tag::alphago_policy_rollout[]
    def policy_rollout(self, game_state):
        for step in range(self.rollout_limit):
            if game_state.is_over():
                break
            move_probabilities = self.rollout_policy.predict(game_state)
            encoder = self.rollout_policy.encoder
            valid_moves = [m for idx, m in enumerate(move_probabilities) 
                           if Move(encoder.decode_point_index(idx)) in game_state.legal_moves()]
            max_index, max_value = max(enumerate(valid_moves), key=operator.itemgetter(1))
            max_point = encoder.decode_point_index(max_index)
            greedy_move = Move(max_point)
            if greedy_move in game_state.legal_moves():
                game_state = game_state.apply_move(greedy_move)

        next_player = game_state.next_player
        winner = game_state.winner()
        if winner is not None:
            return 1 if winner == next_player else -1
        else:
            return 0
# end::alphago_policy_rollout[]


    def serialize(self, h5file):
        raise IOError("AlphaGoMCTS agent can\'t be serialized" +
                       "consider serializing the three underlying" +
                       "neural networks instad.")