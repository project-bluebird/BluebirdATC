import pytest

from bluebird_dt.utility.number import round_nearest


def test_round_nearest():
    """
    Test the functionality of round_nearest():
        - When direction is None.
        - Rounding up.
        - Rounding down.
        - Value error thrown for invalid direction.
    """
    value = [1.11, 2.22, 15.99, 99999.23]
    to = [0.1, 0.1, 1, 0.5]

    # test when direction is None
    none_targets = [1.1, 2.2, 16, 99999]

    for val, amt, tar in zip(value, to, none_targets, strict=False):
        assert round_nearest(val, amt, direction=None) == tar

    # test rounding up
    up_targets = [1.2, 2.3, 16, 99999.5]

    for val, amt, tar in zip(value, to, up_targets, strict=False):
        assert round_nearest(val, amt, direction="up") == tar

    # test rounding down
    down_targets = [1.1, 2.2, 15, 99999]

    for val, amt, tar in zip(value, to, down_targets, strict=False):
        assert round_nearest(val, amt, direction="down") == tar

    # test value error for invalid direction, including that the message names
    # the offending value and the valid options
    invalid_directions = ["this", "is", "wrong", 25]
    for d in invalid_directions:
        with pytest.raises(ValueError, match=rf"Invalid direction given: {d} -- Must be one of"):
            round_nearest(1.1, 0.1, direction=d)
