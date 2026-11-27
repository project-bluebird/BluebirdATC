import math
from decimal import Decimal
from typing import Literal


def round_nearest(num: float, to: float, direction: str | None = None) -> float:
    """
    Rounds floating point number 'num' to the nearest multiple of the
    float 'to', rounded either up or down, based on the given direction. This
    uses a slow, but reliable method via the Decimal class.

    Parameters
    ----------
    num : float
        Number to round.
    to : float
        Multiple of which to round 'num' to be a multiple of.
    direction : str, optional, default = None
        Direction in which to round, valid options are "down" for rounding
        towards negative infinity, "up" for rounding towards positive infinity,
        and None to use the built-in python rounding function.

    Returns
    ----------
    float
        'num' rounded to be a multiple of 'to'

    Notes
        Adapted from https://stackoverflow.com/a/70210770/2161490
    """

    dec_num = Decimal(str(num))
    dec_to = Decimal(str(to))

    if direction is None:
        div = round(dec_num / dec_to)

    elif direction == "up":
        div = math.ceil(dec_num / dec_to)

    elif direction == "down":
        div = math.floor(dec_num / dec_to)

    else:
        raise ValueError(f'Invalid direction given: {direction} -- Must be one of: "up", "down", None')

    return float(div * dec_to)


def which_way(current: float, cleared: float) -> Literal["left", "right"]:
    left = ((current - cleared) + 360) % 360
    right = ((cleared - current) + 360) % 360
    if left < right:
        return "left"
    return "right"
