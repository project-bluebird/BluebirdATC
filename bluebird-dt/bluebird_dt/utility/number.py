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


def which_way(current_heading: float, target_heading: float) -> Literal["left", "right"]:
    """
    Return the shorter way to turn from one heading to another.

    A turn of exactly 180 degrees, or no turn at all, gives "right".

    Parameters
    ----------
    current_heading : float
        The heading turned from, in degrees
    target_heading : float
        The heading turned to, in degrees

    Returns
    ----------
    Literal["left", "right"]
        The direction of the shorter turn
    """

    left = ((current_heading - target_heading) + 360) % 360
    right = ((target_heading - current_heading) + 360) % 360
    if left < right:
        return "left"
    return "right"
