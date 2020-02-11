import os
import numpy as np
from libgoban import IGame



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
    if not g.over():
        g.play(None)
        g.play(None)
    """print(file_name)
    print(str(winner) + " " + points_or_resign)
    print(g.get_winner())"""

    return states, policies, values, player_turn


def SGF_folder_to_dataset(folder_name, out):
    all_states = []
    all_policies = []
    all_values = []
    all_turn = []

    i = 0
    for file_name in os.listdir(folder_name):
        if file_name[-4:] == ".sgf":
            if i % 100 == 0:
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
