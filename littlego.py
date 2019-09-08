#!/bin/python
import click


@click.group()
@click.option("-v", "--verbose", is_flag=True, default=False)
def cli(verbose):
    pass


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default="np_datasets/")
def prepros(path, output):
    from ops import SGF_folder_to_dataset
    import os
    if not os.path.exists(output):
        os.mkdir(output)
    # SGF_folder_rule_filter(sys.argv[1], "Chinese")
    SGF_folder_to_dataset(path, output)
    # SGF_file_to_dataset(sys.argv[1])


@cli.group()
def learn():
    pass


@learn.command()
@click.option("-s", "--board-size", type=click.Choice(["9", "13", "19"]), default="13")
def reinforcement(board_size):
    import ops
    from GoNNAgent import GoNNAgent
    from libgoban import IGame
    # Reinforcement training
    board_size = int(board_size)
    go_agent = GoNNAgent(board_size)
    turn_max = int(board_size ** 2 * 2.)
    for i in range(1000):

        g = IGame(board_size)
        g.display_goban()

        total_turn = 0
        while not g.over():
            print("game {} - total_turn = {}".format(i, total_turn))

            move = go_agent.get_move(g)
            t_move = ops.move_scalar_to_tuple(move, board_size)
            if t_move in g.legals():
                print(t_move)
                g.play(t_move)
            else:
                print("play pass")
                g.play(None)
            g.display_goban()
            total_turn += 1
            if total_turn > turn_max:
                break

        if total_turn > turn_max:
            winner = 2  # draw
            print("(Draw due to maximum moves)")
        else:
            score = g.outcome()
            print(score)
            winner = 2 if score[0] == score[1] else 0 if score[0] > score[1] else 1
        go_agent.end_game(winner)


@learn.command()
@click.argument("path-dataset", type=click.Path(exists=True))
@click.option("-s", "--board-size", default=19, type=click.Choice([9, 13, 19]))
@click.option("-e", "--epoch", default=50000)
@click.option("--report-freq", default=2500)
def supervised(path_dataset, board_size, epoch, report_freq):
    # Supervised training
    from GoNeuralNetwork import GoNeuralNetwork
    from supervised import supervised_training
    neural_network = GoNeuralNetwork(board_size)
    supervised_training(path_dataset, board_size, neural_network, epoch, report_freq)


if __name__ == "__main__":
    cli()
