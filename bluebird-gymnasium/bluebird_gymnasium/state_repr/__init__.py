import numpy as np

from dataclasses import dataclass
from bluebird_gymnasium.utils.module_registry import ModuleRegistry


@dataclass(frozen=True)
class StateReprScaler:
    SCALER_LAT: float = np.pi / 2
    SCALER_LON: float = np.pi
    SCALER_BEARING: float = 2 * np.pi

    SCALER_DIST: int = 50  # nautical miles
    SCALER_DIST_DIFF: int = 20  # nautical miles
    SCALER_CENTRELINE_DIST: int = 50  # nautical miles
    SCALER_EXIT_DIST: int = 50  # nautical miles
    SCALER_INCOMM_DIST: int = 10
    SCALER_OUTCOMM_DIST: int = 10
    SCALER_AC_OTHER_DIST: int = 50
    SCALER_AC_NF_DIST: int = 50
    SCALER_AC_NB_DIST: int = 10
    SCALER_FL: int = 160
    SCALER_FL_DIFF: int = 80
    SCALER_SPEED: int = 200  # nautical miles per hour (knots)
    SCALER_SPEED_DIFF: int = 60  # nautical miles per hour (knots)
    SCALER_VERTICAL_SPEED: int = 1000  # feet per minute
    SCALER_CENTRELINE_DIST_DIFF: int = 4  # nautical miles


@dataclass(frozen=True)
class StateReprClipper:
    CLIP_DIST: int = 150  # nautical miles
    CLIP_DIST_DIFF: int = 60  # nautical miles
    CLIP_INCOMM_DIST: int = 30  # nautical miles
    CLIP_OUTCOMM_DIST: int = 30  # nautical miles
    CLIP_AC_NB_DIST: int = 30  # nautical miles
    CLIP_FL: int = 480  # in flight level
    CLIP_FL_DIFF: int = 240
    CLIP_SPEED: int = 600  # nautical miles per hour (knots)
    CLIP_SPEED_DIFF: int = 180  # nautical miles per hour (knots)
    CLIP_VERTICAL_SPEED: int = 3000  # feet per minute
    CLIP_CENTRELINE_DIST_DIFF: int = 12  # nautical miles


registry_repr = ModuleRegistry()
base_pkg = "bluebird_gymnasium.state_repr"

# full
mod_name = f"{base_pkg}.full"
registry_repr.register("full", f"{mod_name}:FullRepresentation")
registry_repr.register("full_raw", f"{mod_name}:FullRepresentationRaw")

# extra minimal
mod_name = f"{base_pkg}.extraminimal"
registry_repr.register("extra_minimal", f"{mod_name}:ExtraMinimalRepresentation")
registry_repr.register("extra_minimal_raw", f"{mod_name}:ExtraMinimalRepresentationRaw")

# minimal
mod_name = f"{base_pkg}.minimal"
registry_repr.register("minimal", f"{mod_name}:MinimalRepresentation")
registry_repr.register("minimal_raw", f"{mod_name}:MinimalRepresentationRaw")

# relative
mod_name = f"{base_pkg}.relative"
registry_repr.register("relative", f"{mod_name}:RelativeRepresentation")
registry_repr.register("relative_raw", f"{mod_name}:RelativeRepresentationRaw")

# vanilla
mod_name = f"{base_pkg}.vanilla"
registry_repr.register("vanilla", f"{mod_name}:VanillaRepresentation")
registry_repr.register("vanilla_raw", f"{mod_name}:VanillaRepresentationRaw")

# custom drlan
mod_name = f"{base_pkg}.custom.state_repr_drlan"
registry_repr.register("drlan", f"{mod_name}:DrlanRepresentation")
registry_repr.register("drlan_raw", f"{mod_name}:DrlanRepresentationRaw")
