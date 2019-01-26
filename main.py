import random
from enum import Enum
from rusty_goban import IGame


class Color(Enum):
    """
    Color on the goban.
    """
    WHITE = 2,
    BLACK = 1,
    NONE = 0


g = IGame(19)

for i in range(0, 400):
    l = g.legals()
    g.play(random.choice(l))
    g.display()
