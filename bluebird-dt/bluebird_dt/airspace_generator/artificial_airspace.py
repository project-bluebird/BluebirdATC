import typing

from bluebird_dt.airspace_generator import (
    SectorI,
    SectorX,
    SectorXPlus,
    SectorY,
    TwoSectors,
)
from bluebird_dt.airspace_generator.airspace_generator import AirspaceGenerator
from bluebird_dt.core import Airspace, Route
from bluebird_dt.utility.geo_helper import GeoHelper


class ArtificialAirspace(AirspaceGenerator):
    """
    Artificial airspace generator.
    """

    def __init__(
        self,
        sector_type: typing.Literal["i", "x", "y", "two", "xplus"],
        width: int = 20,
        height: int = 60,
        fl_limits: tuple[int, int] = (200, 300),
        alpha: float = 52.5,
        origin: tuple[float, float] = (0.0, 0.0) # rme (-3.533333, 50.716667),
    ):
        """
        Construct a new instance.

        Parameters
        ----------
        sector_type: string
            The requested type of artificial sector (i.e. "i", "x", "y", "two", "xplus").
        width: int
            The width of the artificial sector, in NMI
        height: int
            The height of the artificial sector, in NMI
        fl_limits: tuple[int, int]
            The flight level limits of the airspace, in FL
        alpha: float
            The absolute degrees by which both branches are rotated from a vertical line (0 < alpha < 90). For example,
            alpha=45 results in an X shape with 90 degrees between all branches.
        origin: tuple[float, float]
            The (lon, lat) at which to centre the airspace over. The default is centred on Exeter, UK.

        Raises
        ----------
        ValueError
            Raised if the provided sector_type is unknown (i.e. not "i", "x", "y", "two", "xplus").
        """
        self.sector_type = sector_type
        self.width = width
        self.height = height
        self.fl_limits = fl_limits
        self.alpha = alpha
        self.origin = origin

        # Create the appropriate generator based on sector_type
        match sector_type.lower():
            case "i":
                self.generator = SectorI(width=width, height=height, fl_limits=fl_limits, origin=origin)
            case "x":
                self.generator = SectorX(
                    width=width,
                    height=height,
                    fl_limits=fl_limits,
                    alpha=alpha,
                    origin=origin,
                )
            case "y":
                self.generator = SectorY(
                    width=width,
                    height=height,
                    fl_limits=fl_limits,
                    alpha=alpha,
                    origin=origin,
                )
            case "two":
                self.generator = TwoSectors(width=width, height=height, fl_limits=fl_limits, origin=origin)
            case "xplus":
                self.origin = (0.0, 0.0)
                self.generator = SectorXPlus(origin=self.origin, fl_limits=fl_limits)
            case _:
                raise ValueError(f"Unknown artificial sector type: {sector_type}")

    def generate_airspace(self) -> tuple[Airspace, list[Route]]:
        """
        Generate an Airspace.

        Returns
        ----------
        tuple[Airspace, list[Route]]
            A tuple containing the new airspace object and its corresponding routes.
        """
        # Generate airspace and set up geo_helper
        airspace, routes = self.generator.generate_airspace()
        airspace.geo_helper = GeoHelper(self.origin)

        return airspace, routes
