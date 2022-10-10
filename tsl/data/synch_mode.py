from enum import Enum


class SynchMode(Enum):
    WINDOW = 'window'
    HORIZON = 'horizon'
    STATIC = 'static'


# Aliases
WINDOW = SynchMode.WINDOW
HORIZON = SynchMode.HORIZON
STATIC = SynchMode.STATIC
