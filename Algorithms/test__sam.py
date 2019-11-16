from libgoban import IGame

import MctsAgent

bot = MctsAgent.MCTSAgent(10000, 1.4)

# bot = RandomAgent()

game = IGame(9)

while not game.over():
    move_selected = bot.select_move(game)
    print(move_selected)
    game.display_goban()
    game.play(move_selected)

print(game.outcome())
