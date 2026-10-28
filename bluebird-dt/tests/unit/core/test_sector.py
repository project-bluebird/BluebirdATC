import pytest
import numpy as np

from bluebird_dt.core import Area, Sector, Pos3D, Volume

def test_init(random_adjacent_volumes):
    volumes = random_adjacent_volumes()
    sector = Sector(volumes)
    assert isinstance(sector, Sector)

def test_contains(random_sector, random_pos3d):
     sector = random_sector()
     point = random_pos3d()
     expect_contains = False
     # see if the point is within the sector's volumes
     for vol in sector.volumes:
         expect_contains |= vol.contains(point)
     assert sector.contains(point) == expect_contains


def test_boundary(random_sector):
    sector = random_sector()
    area = sector.boundary()
    assert isinstance(area, Area)
    


def test_contains_laterally(random_sector, random_pos2d):
    sector = random_sector()
    point = random_pos2d()
    area = sector.boundary()
    assert sector.contains_laterally(point) == area.contains(point)


def test_get_bounds(random_sector):
    sector = random_sector()
    bounds = sector.get_bounds()
    assert isinstance(bounds, tuple)
    assert len(bounds) == 2
    for b in bounds:
        assert isinstance(b, np.ndarray)
        assert b.shape == (3,)


@pytest.mark.parametrize(
    "volume_limits, fl_limits, expected",
    [
        ([(215, 660)], (50, 400), range(220, 410, 10)),
        ([(45, 395)], (50, 400), range(50, 400, 10)),
        ([(60, 400)], (55, 395), range(60, 400, 10)),
        # the first Volume starts at FL215, but the Sector at FL45
        ([(215, 660), (45, 215)], (50, 400), range(50, 410, 10)),
    ],
)
def test_allowed_flight_levels(random_area, volume_limits, fl_limits, expected):
    """
    The levels should be the multiples of 10 within both fl_limits and the range of all the Sector's Volumes.
    """
    area = random_area()
    sector = Sector([Volume(area, min_fl, max_fl) for min_fl, max_fl in volume_limits])
    assert list(sector.allowed_flight_levels(fl_limits)) == list(expected)


def test_no_allowed_flight_levels(random_area):
    """
    fl_limits outside the Sector's range should raise an error giving the range and the limits.
    """
    sector = Sector([Volume(random_area(), 60, 400)])
    with pytest.raises(ValueError, match=r"FL60-FL400.*\(410, 500\)"):
        sector.allowed_flight_levels((410, 500))


def test_centre(random_sector):
    sector = random_sector()
    centre = sector.centre
    assert isinstance(centre, Pos3D)
    # centre must be within the sector
    assert sector.contains(centre)


def test_distance(random_sector, random_pos2d):
    sector = random_sector()
    point = random_pos2d()
    distance = sector.distance(point)
    assert isinstance(distance, float)
    assert distance > 0.
