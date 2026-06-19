import random
import string

import pytest

from bluebird_dt.utility.sample import score_from_rank

def test_score_from_rank():
    """
    Test the functionality of score_from_rank():
        - Assuming standard normal distribution
        - Assuming a truncated normal distribution
        - Value error thrown if percentage rank is outside (0, 100).
    """

    # Check the routine returns sensible scores for a range on input combinations
    percentile_rank_values = [None, 25.0, 50.0, 25.0, 75.0, 25.0, 75.0, 50.0]
    nominal_values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    standard_deviation_values = [1.0, None, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    min_score_values = [None, None, None, None, None, 0.4, None, 0.2]
    max_score_values = [None, None, None, None, None, None, 1.5, 1.8]
    returned_scores = [1.0, 1.0, 1.0, 0.3255102, 1.6744898, 0.8887015, 1.0466323, 1.0]

    for percentile_rank, nominal, standard_deviation, min_score, max_score, score in zip(
        percentile_rank_values,
        nominal_values,
        standard_deviation_values,
        min_score_values,
        max_score_values,
        returned_scores,
        strict=False,
    ):
        assert score_from_rank(
            percentile_rank=percentile_rank,
            nominal=nominal,
            standard_deviation=standard_deviation,
            min_score=min_score,
            max_score=max_score,
        ) == pytest.approx(score)

    # Check the routine returns an exception if the provided percentile rank
    # is outside the range (0, 100)
    percentile_rank_values = [0.0, 100.0, -1.0, 115.0]
    for percentile_rank in percentile_rank_values:
        with pytest.raises(ValueError):
            score_from_rank(percentile_rank=percentile_rank, nominal=1.0, standard_deviation=1.0)

    # Check the routine returns an exception if the provided max score is less that the
    # min score
    with pytest.raises(ValueError):
        score_from_rank(percentile_rank=25.0, nominal=1.0, standard_deviation=1.0, min_score=2.0, max_score=1.5)
