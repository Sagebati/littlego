import random
import rusty_goban as game

g = game.IGame(19)

for i in range(0, 10):
    l = g.legals()
    g.play(random.choice(l))
    g.display()
