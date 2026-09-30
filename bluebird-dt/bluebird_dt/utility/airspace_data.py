from __future__ import annotations

import json
import os
import typing
from pathlib import Path

import geojson
import pandas as pd

if typing.TYPE_CHECKING:
    from bluebird_dt.core import Sector

from bluebird_dt.core import Airspace, Airway, Area, Fixes, Volume
from bluebird_dt.logger import logger


def create_airspace(
    sector_geojson_paths: list[str],
    fixes_path: list[str],
    airways_path: str | None = None,
) -> Airspace:
    """
    Creates an Airspace object from sector, fixes, and optional airway data files.

    Loads sectors from GeoJSON files, fixes from CSV or GeoJSON files, and optionally airways from a JSON file.

    Parameters
    ----------
    sector_geojson_paths : list[str]
        List of paths to sector GeoJSON files. Sector names are currently derived from the last two characters of the
        filename stem (e.g., "..._31.geojson"). Ensure filenames follow this convention.
    fixes_paths : list[str]
        List of paths to fixes files (supports .csv, .gz, .geojson). Fixes from all files will be merged.
    airways_path : str | None, optional
        Path to a JSON file containing airway definitions. If None, no airways are loaded. Defaults to None.
    default_airway_min_fl : int, optional
        Default minimum flight level to use for airways loaded from JSON if not specified otherwise. Defaults to 0.
    default_airway_max_fl : int, optional
        Default maximum flight level to use for airways loaded from JSON if not specified otherwise. Defaults to 660.

    Returns
    -------
    Airspace
        An Airspace object containing the loaded sectors, fixes, and airways.
    """

    # initialise dictionary to store all [sector_name -> Sector] combinations
    sectors = {}

    # fill the dictionary by reading from all files (sector name is the file name)
    for sector_geojson_path in sector_geojson_paths:
        # TODO: Re-evaluate sector naming logic. Is filename reliable?
        # get filename without extension, take last 2 chars as sector name
        sector_name = Path(sector_geojson_path).stem[-9:]
        if not sector_name:
            logger.warning(
                f"Could not derive sector name from filename: {sector_geojson_path}",
                stacklevel=2,
            )
            continue

        try:
            sectors[sector_name] = create_sector(sector_geojson_path=sector_geojson_path)
        except FileNotFoundError:
            logger.warning(f"Sector file not found: {sector_geojson_path}", stacklevel=2)
        except Exception as e:
            raise e
            logger.warning(
                f"Failed to load sector {sector_name} from {sector_geojson_path} because: {e}",
                stacklevel=2,
            )

    # create a Fixes object out of a file
    fixes: Fixes = Fixes({})

    for path in fixes_path:
        try:
            new_fixes = load_fixes(fixes_path=path)
            fixes = new_fixes | fixes
        except FileNotFoundError:  # noqa: PERF203
            logger.warning(f"Fixes file not found: {path}", stacklevel=2)
        except Exception as e:
            logger.warning(f"Failed to load fixes from {path} because: {e}", stacklevel=2)

    if airways_path is not None:
        with open(airways_path) as f:
            airway_data = json.load(f)

        airways = {}

        for key, val in airway_data.items():
            airways[key] = Airway.from_list_of_fixes(key, val["fixes"], 0, 660, fixes)

    else:
        airways = None

    # create an Airspace and return it
    return Airspace(sectors=sectors, fixes=fixes, airways=airways)


def load_volumes(geojson_path: str) -> list[Volume]:
    """
    Loads volumes defined in a GeoJSON file into a list of Volume objects.

    Reads a GeoJSON file, expecting a FeatureCollection where each feature represents a volume.
    For each feature, it extracts the geometry (polygon coordinates), minimum and maximum flight levels
    (from 'base' and 'top' properties, handling 'SFC'), and optional metadata ('sector', 'description',
    'airspace_id') from the properties.

    Parameters
    ----------
    geojson_path : str
        Path to the GeoJSON file containing volume features.

    Returns
    -------
    list[Volume]
        A list of Volume objects, each representing a volume defined in the GeoJSON file.
    """
    from bluebird_dt.core import Pos2D, Volume
    print(geojson_path)
    with open(geojson_path) as sector_geojson_file:
        sector_dict_full = json.load(sector_geojson_file)

    volume_list: list[Volume] = []

    # construct volumes from the geojson feature collection
    for feature in sector_dict_full["features"]:
        # grab the bits we need
        props: dict[str, str] = feature["properties"]
        coords: list[list[tuple[float, float]]] = feature["geometry"]["coordinates"]

        # grab the extra meta data, if present
        sector_name = props.get("sector")
        description = props.get("description")
        airspace_id = props.get("airspace_id")
        print(props)

        # extract the flight levels. note #SFC = SurFaCe (height = 0), otherwise it is of form e.g. FL250
        min_fl = 0 if props["base"] == "SFC" else int(float(props["base"][2:]))
        max_fl = int(float(props["top"][2:]))

        # extract the area coordinates; stored in the first element of the coordinates list
        area_coordinates = [Pos2D(coordinate[1], coordinate[0]) for coordinate in coords[0]]
        area = Area(area_coordinates)

        volume = Volume(area, min_fl, max_fl, sector_name, description, airspace_id)
        volume_list.append(volume)

    return volume_list


def create_sector(sector_geojson_path: str) -> Sector:
    """
    Constructs a Sector object from sector GeoJSON files.

    This function loads the primary volumes defining a sector from the specified GeoJSON file.
    It then checks for the existence of corresponding Area of Responsibility (AoR) and conditional volume files,
    conventionally named by appending "_aor" and "_conditional" respectively to the base filename
    (e.g., "sector_name_aor.geojson", "sector_name_conditional.geojson"). If these files exist, their volumes are
    loaded. Conditional volumes are processed to ensure their keys (fix pairs) are sorted alphabetically.

    Parameters
    ----------
    sector_geojson_path : str
        The path to the primary sector GeoJSON file.

    Returns
    -------
    Sector
        A `Sector` object representing the airspace defined in the GeoJSON file(s), including primary volumes,
        AoR volumes (if found), and conditional volumes (if found).
    """
    from bluebird_dt.core import Sector

    # load the sector volumes from the geojson file
    sector_volumes = load_volumes(sector_geojson_path)

    # check if the area of responsibility file exists and load it if it does
    if os.path.exists(aor_path := sector_geojson_path.replace(".geojson", "_aor.geojson")):
        area_of_responsibility = load_volumes(aor_path)
    else:
        area_of_responsibility = None

    # check if the conditional volumes file exists and load it if it does
    if os.path.exists(cond_path := sector_geojson_path.replace(".geojson", "_conditional.geojson")):
        named_volumes = load_volumes(cond_path)

        # preprocess the conditional volumes to ensure they are in the correct format
        conditional_volume_dict = create_conditional_route_dict(named_volumes)
    else:
        conditional_volume_dict = None

    return Sector(sector_volumes, area_of_responsibility, conditional_volume_dict)


def load_fixes(fixes_path: str) -> Fixes:
    """
    Loads all fixes and coordinates from file on disk, and returns a Fixes object.
    See specific documentation for file type.

    Implemented file types:
        - csv: load_fixes_csv
        - gzip compressed csv: load_fixes_csv (used for fixes_combined.csv.gz)
        - geojson:  load_from_geojson

    Parameters
    ----------
    fixes_path: str
        A path to the fixes file of an implemented file type.

    Returns
    -------
    Fixes
        A fixes object containing all the fixes and their coordinates
    """
    _, file_extension = os.path.splitext(fixes_path)

    match file_extension:
        case ".csv" | ".gz":
            return load_fixes_csv(fixes_path)
        case ".geojson":
            return load_from_geojson(fixes_path)
        case _:
            raise NotImplementedError(
                f"File type {file_extension} for loading fixes for file {fixes_path} is not supported."
                f"BluebirdATC currently only supports csv and geojson."
            )


def load_from_geojson(fixes_path: str) -> Fixes:
    """
    Loads all fixes and coordinates from .geojson file on disk, and returns a Fixes object.
    The geojson file is expected as a Feature Collection containing Points.

    Note the Pos2D constructor used does not perform coordinate transformations, therefore assuming they
    are already in WGS84.

    Parameters
    ----------
    fixes_path: str
        A path to the fixes geojson file.

    Returns
    -------
    Fixes
        A fixes object containing all the fixes and their coordinates
    """
    with open(fixes_path) as f:
        tree = geojson.loads(f.read())

    return Fixes.from_geojson(tree)


def load_fixes_csv(fixes_path: str) -> Fixes:
    """
    Loads all fixes and coordinates from .csv file on disk, and returns a Fixes object.

    Parameters
    ----------
    fixes_path: str
        A path to the combined fixes csv file.

    returns
    ----------
    fixes
        a fixes object containing all the fixes and their coordinates.
    """

    from bluebird_dt.core import Fixes, Pos2D

    fix_dtypes = {
        "fix": "string",
        "lat": "float64",
        "lon": "float64",
        "visible": "string",
    }
    # Read the dataframe
    df = pd.read_csv(fixes_path, dtype=fix_dtypes).filter(items=["fix", "lat", "lon", "visible"])

    # Convert 'visible' column to boolean
    visibility_dict = None
    if "visible" in df.columns:
        df["visible"] = df["visible"].map({"Y": True, "N": False})
        visibility_dict = dict(zip(df["fix"], df["visible"], strict=False))

    # Drop duplicate fix names (keep the first entry) and reset the index
    df = df.drop_duplicates(subset=["fix"], keep="first").reset_index()
    df = df.drop(columns=["index"])
    df["positions"] = pd.Series(
        [Pos2D(lat, lon) for lat, lon in zip(df["lat"], df["lon"], strict=False)],
        index=df.index,
    )
    fixes_dict = dict(zip(df["fix"], df["positions"], strict=False))

    return Fixes(fixes_dict, visibility_dict)


def create_conditional_route_dict(named_volumes: list[Volume]) -> dict[str, Volume]:
    """
    Processes a list of Volumes intended for use as conditional route volumes.

    Each Volume in the input list is expected to have a `description` attribute containing two fix names separated by
    an underscore (e.g., "FIXB_FIXA"). This function extracts these fix names, sorts them alphabetically, joins them
    back with an underscore to create a standardized string key (e.g., "FIXA_FIXB"), and maps this key to the
    corresponding Volume object in the output dictionary.

    Parameters
    ----------
    named_volumes : list[Volume]
        A list of Volume objects, each expected to represent a conditional volume with its associated fix pair defined
        in the `description` attribute.

    Returns
    -------
    dict[str, Volume]
        A dictionary mapping alphabetically sorted fix-pair strings (e.g., "FIXA_FIXB") to their corresponding
        Volume objects.

    Raises
    ------
    ValueError
        If the descriptions of the input volumes result in duplicate sorted string keys after processing.

    Warns
    -----
    UserWarning
        - If a Volume in `named_volumes` has a `description` attribute set to None.
        - If a Volume's `description` does not contain exactly two parts when split by '_'.
    """

    conditional_volume_dict: dict[str, Volume] = {}

    for volume in named_volumes:
        description = volume.description

        if description is None:
            logger.warning(
                f"Volume '{volume}' does not have a description. "
                "This volume will not be included in the conditional route dictionary.",
                stacklevel=2,
            )
            continue

        fix_pair = description.split("_")
        if len(fix_pair) != 2:
            logger.warning(
                f"Volume '{volume}' description '{description}' does not contain exactly two fixes."
                " This volume will not be included in the conditional route dictionary.",
                stacklevel=2,
            )
            continue

        # sort the fix pair alphabetically to create a standardised key
        key = "_".join(sorted(fix_pair))

        # Check for duplicate keys during insertion for a more informative error
        if key in conditional_volume_dict:
            logger.warning(
                f"Duplicate key '{key}' found for volumes '{conditional_volume_dict[key]}' and '{volume}'. "
                "This may lead to unexpected behaviour.",
                stacklevel=2,
            )

        conditional_volume_dict[key] = volume

    return conditional_volume_dict
