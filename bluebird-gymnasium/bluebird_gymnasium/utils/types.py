import numpy as np
import typing
from typing import TypeAlias
from dataclasses import dataclass
from enum import Enum, EnumMeta, IntEnum

from bluebird_dt.core.pos2d import Pos2D
from bluebird_dt.core.pos4d import Pos4D
from bluebird_dt.core.coordination import Coordination

# type alias
Number: typing.TypeAlias = typing.Union[int, float]
Line: typing.TypeAlias = tuple[Pos2D, Pos2D]


####### enums
class MetaEnum(EnumMeta):
    # support for python < 3.12
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class StrEnum(str, Enum):
    # support for python < 3.11
    @staticmethod
    def _generate_next_value_(name, start, count, last_values) -> str:
        return name.lower()

    def __str__(self) -> str:
        return self.value


class PositionStatus(IntEnum, metaclass=MetaEnum):
    IN_SECTOR = 1
    OUT_SECTOR = 2
    EXIT_REACHED = 3
    BEFORE_ENTRY = 4


class InteractionDistance(IntEnum, metaclass=MetaEnum):
    CLOSER = -1
    MAINTAIN = 0
    FARTHER = 1


class InteractionCategory(IntEnum, metaclass=MetaEnum):
    NONE = 0
    SAME_TRACK = 1
    OPPOSITE_TRACK = 2
    CROSS_TRACK_LEFT = 3
    CROSS_TRACK_RIGHT = 4


class LineIntersection(IntEnum, metaclass=MetaEnum):
    NONE = 0
    EXIST = 1
    PROBABLE = 2


class Quadrant(IntEnum, metaclass=MetaEnum):
    Q1 = 1
    Q2 = 2
    Q3 = 3
    Q4 = 4


class TurnDirection(IntEnum, metaclass=MetaEnum):
    LEFT = -1
    RIGHT = 1
    NO_TURN = 0


class InteractionRelevance(IntEnum, metaclass=MetaEnum):
    UNDEFINED = 0
    LEVEL_1 = 1  # primary interactions
    LEVEL_2 = 2  # secondary interactions


####### data classes
@dataclass
class ACPositionInfo:
    position_status: PositionStatus
    incomm_status: bool
    outcomm_status: bool
    dist_to_sector_entry: float
    dist_away_from_sector_exit: float
    dist_away_from_incorrect_sector_exit: float
    incorrect_exit_position: Pos2D | None


@dataclass
class IntersectionInfo:
    status: bool
    location: Pos2D | None = None
    location_name: str | None = None


@dataclass
class InteractionInfo:
    other_callsign: str
    dist_ac_other: float
    bearing_ac_other: float
    angle_diff_ac_other: float  # current heading
    turn_dir_ac_other: int
    angle_diff_ac_other_sh: float  # selected heading (sh)
    turn_dir_ac_other_sh: int
    dist_type_ac_other: InteractionDistance
    track_category_ac_other: InteractionCategory
    fl_diff_ac_other: float
    selected_fl_diff_ac_other: float
    speed_diff_ac_other: float
    centreline_dist_diff_fr: float
    centreline_dist_diff_cr: float
    lateral_dist_thresh_sv: float  # sv: safety violation
    relevance: InteractionRelevance = InteractionRelevance.UNDEFINED
    intersections: list[IntersectionInfo] | None = None
    main_intersection: IntersectionInfo | None = None
    proxy_intersection: IntersectionInfo | None = None


@dataclass(frozen=True)
class MinAircraftSeparation:
    LATERAL: float = 5.0  # in nautical miles
    VERTICAL: float = 10.0  # in flight levels


@dataclass
class ForwardFixesInfo:
    num_fixes: int | None = None
    use_filed_route: bool = True


@dataclass
class ACStateTracker:
    # static data
    sector_entry_pos: Pos2D | None = None
    sector_exit_pos: Pos2D | None = None
    sector_exit_window: tuple[Pos2D, Pos2D] | None = None
    sector_entry_timestep: int | None = None
    entry_coords: dict[str, Coordination] | None = None
    exit_coords: dict[str, Coordination] | None = None

    # dynamic data: first initialize
    pos_status: PositionStatus | None = None
    incomm_status: bool | None = None
    outcomm_status: bool | None = None
    dist_to_sector_entry: float | None = None
    dist_away_from_sector_exit: float | None = None
    dist_away_from_incorrect_sector_exit: float | None = None
    incorrect_sector_exit_pos: Pos2D | None = None
    # safety debug: updated in safety rewards
    safety_debug: dict[str, dict[str, typing.Any]] | None = None
    steps_since_action: int | None = None
    pos_at_last_route_direct: Pos2D | None = None

    # dynamic data: update
    centreline_info_fr: tuple[float, int, float] | None = None  # filed route
    centreline_info_cr: tuple[float, int, float] | None = None  # current route
    curr_sector: str | None = None
    future_trajectory: list[Pos4D] | None = None
    previous_fix_fr: str | None = None  # filed route (fr)
    next_fix_fr: str | None = None  # filed route (fr)
    previous_fix_cr: str | None = None  # current route (cr)
    next_fix_cr: str | None = None  # current route (cr)
    flight_level: float | None = None
    selected_flight_level: float | None = None
    heading: float | None = None
    selected_heading: float | None = None
    position: Pos2D | None = None
    speed_tas: float | None = None
    speed_ground: float | None = None
    selected_speed_cas: float | None = None
    vertical_speed: float | None = None
    linear_dist_to_exit: float | None = None
    track_dist_to_exit_fr: float | None = None
    track_dist_to_exit_cr: float | None = None
    dist_to_target_fl: float | None = None

    nearest_360_boundary_pos: Pos2D | None = None
    nearest_360_boundary_dist: float | None = None
    nearest_360_boundary_bear: float | None = None  # bearing
    # depends on next_fix_cr and track_dist_to_exit_cr
    nearest_forward_boundary_pos: Pos2D | None = None
    nearest_forward_boundary_dist: Number | None = None

    step_counter: int | None = None
    action: int | None = None

    # lazily set: used during traffic monitor updates in interaction utils
    # if extra control points beyond the ones `future_trajectory` are needed.
    # then they're cached for reuse within a single traffic monitor update.
    # at the end of the update, the cache is cleared.
    extra_future_trajectory: dict[str, list[Pos4D]] | None = None


class NamedLine:
    line: Line
    name: tuple[str, str]

    def __init__(self, line: Line, name: tuple[str | None, str | None]):
        self.line = line
        self.name = name

    def __getitem__(self, idx: int) -> Pos2D:
        return self.line[idx]

    def start_position(self) -> Pos2D:
        return self.line[0]

    def end_position(end) -> Pos2D:
        return self.line[1]

    def start_position_name(self) -> str | None:
        return self.name[0]

    def end_position_name(end) -> str | None:
        return self.name[1]

    def get_line(self) -> Line:
        return self.line

    def get_name(self) -> tuple[str | None, str | None]:
        return self.name

    def length(self) -> Number:
        return self.line[0].distance(self.line[1])


@dataclass
class Path:
    segments: list[NamedLine]


####### extras
QuadrantBound: dict[Quadrant, tuple[float, float]] = {
    Quadrant.Q1: (0.0, 90.0),
    Quadrant.Q2: (90.0, 180.0),
    Quadrant.Q3: (180.0, 270.0),
    Quadrant.Q4: (270.0, 360.0),
}
