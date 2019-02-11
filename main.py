import random
from enum import Enum
from libshusaku import IGame


class Color(Enum):
    """
    Color of the stones in the goban.
    """
    WHITE = 2,
    BLACK = 1,
    NONE = 0


class Player(Enum):
    """
    Color of the player
    """
    White = True,
    Black = False,


for i in range(0, 500):
    l = g.legals()
    g.play(random.choice(l))
    print(g.goban())
    g.display()
