from __future__ import annotations

import numpy as np
from scipy.stats import truncnorm


def score_from_rank(
    percentile_rank: float | None,
    nominal: float,
    standard_deviation: float | None,
    min_score: float | None = None,
    max_score: float | None = None,
) -> float:
    """
    Returns the score that corresponds to a given percentile rank for a truncated normal distribution with
    a given mode and standard deviation

    Parameters
    ----------
    percentile_rank: float
        The percentile rank of the score
    nominal: float
        The mode/median/mean of the normal distribution
    standard_deviation: float
        The standard deviation of the normal distribution
    min_score: float
        The minimum score in the truncated norm
    max_score: float
        The maximum score in the truncated norm

    Returns
    -------
    Float
        The score corresponding to the given percentile rank
    """
    if percentile_rank is None or standard_deviation is None:
        return nominal

    if percentile_rank > 0.0 and percentile_rank < 100.0:
        if min_score is not None and max_score is not None and min_score >= max_score:
            raise ValueError("The input max_score must be larger than min_score")
        a = (min_score - nominal) / standard_deviation if min_score is not None else -np.inf
        b = (max_score - nominal) / standard_deviation if max_score is not None else np.inf

        return nominal + standard_deviation * float(truncnorm.ppf(percentile_rank / 100.0, a, b))

    raise ValueError(f"Percentile rank value of {percentile_rank} is outside the allowed range (0, 100)")


def apply_rocd_uncertainty(
    nominal_vertical_speed: float, rocd_uncertainty: dict[str, float], percentile_rank: float | None
) -> float:
    """
    If percentile rank has been specified, use the uncertainty data to draw a vertical speed score
    from the vertical speed probability distribution.
    """
    sd = rocd_uncertainty["sigma"]

    min_score = (
        None
        if rocd_uncertainty["minimum"] is None
        # Else apply shift to the observed minimum score since we are using the observed speed
        # distribution but with a mean/mode equal to the nominal speed value obtained from
        # the speed profile data (e.g. from BADA).
        else nominal_vertical_speed - rocd_uncertainty["norm_mean"] + rocd_uncertainty["minimum"]
    )

    max_score = (
        None
        if rocd_uncertainty["maximum"] is None
        # Else apply shift to the observed maximum score since we are using the observed speed
        # distribution but with a mean/mode equal to the nominal speed value obtained from
        # the speed profile data (e.g. from BADA).
        else nominal_vertical_speed - rocd_uncertainty["norm_mean"] + rocd_uncertainty["maximum"]
    )

    return score_from_rank(
        percentile_rank,
        nominal=nominal_vertical_speed,
        standard_deviation=sd,
        # Limiting the vertical speed here to a minimum of 500 ft/min
        min_score=min_score if min_score is not None and min_score > 500.0 else 500.0,
        max_score=max_score,
    )


def apply_speed_uncertainty(
    nominal_speed: float, speed_uncertainty: dict[str, float], percentile_rank: float | None
) -> float:
    sd = speed_uncertainty["sigma"]

    min_score = (
        None
        if speed_uncertainty["minimum"] is None
        # Else apply shift to the observed minimum score since we are using the observed speed
        # distribution but with a mean/mode equal to the nominal speed value obtained from
        # the speed profile data (e.g. from BADA).
        else nominal_speed - speed_uncertainty["norm_mean"] + speed_uncertainty["minimum"]
    )

    max_score = (
        None
        if speed_uncertainty["maximum"] is None
        # Else apply shift to the observed maximum score since we are using the observed speed
        # distribution but with a mean/mode equal to the nominal speed value obtained from
        # the speed profile data (e.g. from BADA).
        else nominal_speed - speed_uncertainty["norm_mean"] + speed_uncertainty["maximum"]
    )

    return score_from_rank(
        percentile_rank,
        nominal=nominal_speed,
        standard_deviation=sd,
        min_score=min_score,
        max_score=max_score,
    )
