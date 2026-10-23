import json
import math
import os
import re
import time
import typing
import warnings
from collections import deque
from collections.abc import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from bluebird_dt.core import (
    Aircraft,
    Airspace,
    Area,
    Environment,
    Fixes,
    Pos2D,
    Pos3D,
    Pos4D,
    Route,
    Sector,
    Volume,
)
from bluebird_dt.core.action import Action
from bluebird_dt.render.render import Render
from bluebird_dt.utility.geo_helper import GeoHelper
from bluebird_dt.utility.stereographic_projection import lon_lat_to_x_y_km, lon_lat_to_x_y_nm

DEFAULT_MARKER_SIZE = mpl.rcParams["lines.markersize"]
DEFAULT_FONT_SIZE = mpl.rcParams["font.size"]  # usually 10.0
DEFAULT_FONT_WIDTH = mpl.rcParams["font.stretch"]  # width of each character
DEFAULT_PLAN_COLOR = (1.0, 0.0, 1.0, 1.0)  # rgba: magenta


class Plan(typing.NamedTuple):
    positions: list[Pos3D]

    @staticmethod
    def from_json(s: str) -> "Plan":
        data = json.loads(s)
        return Plan([Pos3D.from_str(pos3d) for pos3d in data])


class Radar(Render):
    """
    Radar image generator.
    """

    def __init__(
        self,
        centre: Pos2D,
        scale: float,
        aspect_ratio: float,
        aircraft_scale: float = 1.0,
        render_fixes: bool | list = True,
        render_sys_fixes: bool = False,
        render_routes: bool = True,
        render_sep_bound: bool = False,
        aircraft_lateral_sep: float = 5.0,
        auto_display: bool = True,
        blips: int = 5,
        sector_name: str | None = None,
        show_spines: bool = True,
        display_units: str | Callable = "lonlat",
        scaled: bool = True,
        display_actions: bool = False,
    ):
        """
        Initialise the Radar to observe a central location and a given width scale [nmi] with a given
        aspect-ratio (=width/height).

        Parameters
        ----------
        centre: Pos2D
            Required parameter, centre of the environment
        scale: float
            Required parameter, width scale for the images in nautical miles in whatever units are
            being used for display. See display_units.
        aspect_ratio: float
            Required parameter, width/height for output image
        aircraft_scale: float
            Required parameter, relative scale factor of aircraft
            default = 1.0
        render_fixes: bool or list[str]
            Required parameter, whether to render fixes, or if list is provided, render only fixes given.
            if a fix is not within the display area (i.e., defined by the radar scale/width), then it will
            not be displayed. To display such a fix, increase the radar scale (width in nautical
            miles or aspect ratio to increase the height).
            default = True
        render_sys_fixes: bool
            Required parameter, whether to render system fixes. Note: system fixes will NOT be rendered if
            render_fixes=False regardless of value
            default = False
        render_routes: bool
            Required parameter, whether to render routes
            default = True
        render_sep_bound: bool
            Required parameter, whether to render separation bound for each aircraft
            default = False
        aircraft_lateral_sep: float
            Required parameter, the radius of the lateral separation bound for each aircraft
            default = 5.0
        auto_display: bool
            Required parameter, whether to display the simulation after drawing it in a buffer.
            if it is set to False, the caller is responsible for displaying the plot (e.g., via
            `plt.show(...)` or flushing and drawing the figure's canvas as was employed in
            the `.draw(...)` method in this class.)
            Default = True
        blips: int
            Required parameter, number of blips trailing the aircraft
            Default = 5
        sector_name: str or None
            The name of the sector being displayed. This is useful to set in order to retrieve the
            correct exit flight level (from coordination data) for each aircraft being displayed
        show_spines: bool
            Show the plot scales.
            Default = False.
        display_units: str | Callable
            Set the units of what will be displayed.
            "lonlat" means degrees longitude (horizontally) and latitude (vertically);
            "nm" means nautical miles with the origin at the NATS standard projection.
            "km" means kilometres with the origin at the NATS standard projection.
            Specifying a function x, y = func(lon, lat) that maps a (lon, lat) coordinate to x and y
            allows positions to be displayed in any format.
        scaled: bool
            Use axis('scaled') to size the figure to equal units in both coordinate directions.
            Default is True.
        display_actions: bool
            Show actions in plot.
            Default = False.
        """

        if isinstance(display_units, Callable):
            self.lonlat_to_xy = display_units
            user_defined = True
        else:
            user_defined = False
            match display_units:
                case "lonlat":
                    self.lonlat_to_xy = lambda lon, lat: (lon, lat)
                case "nm":
                    self.lonlat_to_xy = lon_lat_to_x_y_nm
                case "km":
                    self.lonlat_to_xy = lon_lat_to_x_y_km
                case _:
                    raise ValueError("Unknown value for display_units:", display_units)

        if aspect_ratio <= 0.0:
            raise ValueError("Aspect ratio must be positive.")

        self.centre = centre
        self.centre_xy = np.asarray(self.lonlat_to_xy(centre.lon, centre.lat))
        self.scale = scale
        self.aspect_ratio = aspect_ratio
        self.scaled = scaled
        self.display_actions = display_actions
        self.aircraft_scale = self.scale * aircraft_scale * 1e-4
        self.label_lat_diff = self.scale * aircraft_scale * 2e-4
        self.label_spacing = self.scale * aircraft_scale * 3e-4

        if isinstance(render_fixes, list):
            self.render_fixes = True
            self.render_fixes_list = render_fixes
        else:
            self.render_fixes = render_fixes
            self.render_fixes_list = None

        self.render_fixes = render_fixes
        self.render_sys_fixes = render_sys_fixes
        self.render_routes = render_routes
        self.render_sep_bound = render_sep_bound

        self.aircraft_lateral_sep = aircraft_lateral_sep
        self.auto_display = auto_display
        self.blips = blips
        self.sector_name = sector_name

        self.counter = 0

        gh = GeoHelper()
        west_lon, west_lat = gh.forward(self.centre.lon, self.centre.lat, distance=self.scale / 2.0, heading=270.0)
        east_lon, east_lat = gh.forward(self.centre.lon, self.centre.lat, distance=self.scale / 2.0, heading=90.0)

        west_x, west_y = self.lonlat_to_xy(west_lon, west_lat)
        east_x, east_y = self.lonlat_to_xy(east_lon, east_lat)

        self.width = east_x - west_x
        # Check that the (user-defined) transformation hasn't swapped x and y
        if user_defined and math.fabs(self.width) < 1e-3 and math.fabs(east_y - west_y) > 1e-3:
            self.width = east_y - west_y

        self.height = self.width / self.aspect_ratio

        self.min_x = self.centre_xy[0] - (self.width / 2.0)
        self.min_y = self.centre_xy[1] - (self.height / 2.0)
        self.max_x = self.centre_xy[0] + (self.width / 2.0)
        self.max_y = self.centre_xy[1] + (self.height / 2.0)

        self.traces_dict = {}
        self.action_queue = deque(maxlen=5)
        # initialise queue with empty string.
        for _ in range(self.action_queue.maxlen):
            self.action_queue.append("")

        # matplotlib-related attributes
        self.figure = None
        self.plot_width = 9  # in inches
        self.plot_height = self.plot_width / self.aspect_ratio
        self.show_spines = show_spines

    def reset_step_counter(self):
        self.counter = 0

    def get_figure(self) -> Figure:
        """
        Get the maplotlib figure for the radar visualisation.

        If it hasn't been created, then create it.

        Returns
        -------
        figure :
            the matplotlib figure.
        """

        plt.ion()  # interactive mode

        if self.figure is None:
            self.figure, ax = plt.subplots(figsize=(self.plot_width, self.plot_height))

            ax.set_xlim(self.min_x, self.max_x)
            ax.set_ylim(self.min_y, self.max_y)
            if self.scaled:
                ax.axis("scaled")

            # remove the visibility of the axes boundaries.
            if not self.show_spines:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)

                ## remove the visibility of ticks in axes.
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

        else:
            ax = self.figure.get_axes()[0]

            ax.set_xlim(self.min_x, self.max_x)
            ax.set_ylim(self.min_y, self.max_y)

        # set the figure as the current (global) figure.
        # i.e. the figure returned when `plt.gcf()` is called
        _ = plt.figure(self.figure)

        return self.figure

    def save(self, filename: str):
        """
        Save the drawing to an image or pdf then clear the figure.

        Parameters
        ----------
        filename: str
            Required parameter, filename for output. formats supported include:
            png, jpg, svg.
            png is used by default if no format is defined via the filename.
        """

        _, file_extension = os.path.splitext(filename)
        if file_extension == "":
            # defaults to png
            file_extension = ".png"
            filename += file_extension

        figure = self.get_figure()
        figure.savefig(filename, dpi=256)

    def clear_screen(self):
        """Clear the radar screen."""

        if self.figure is not None:
            # clear any previous drawings
            ax = self.figure.get_axes()[0]
            ax.clear()

    def reset(self):
        """Reset the radar."""

        self.reset_step_counter()
        self.clear_screen()

    def draw(self, environment: Environment, actions_log: list[Action] | None = None) -> tuple[Figure, Axes]:
        """
        Draw the Environment.

        Parameters
        ----------
        environment: Environment
            Required parameter, the environment
        actions_log: list[Action] | None
            Optional parameter, the log of actions
            Default = None

        Returns
        -------
        tuple :
            two elements, the first being a matplotlib figure and the second is the
            matplotlib axes connected to figure (the buffer where the plot is drawn).
        """

        self.clear_screen()
        figure = self.get_figure()
        ax = figure.get_axes()[0]

        self.draw_environment(environment, actions_log=actions_log)

        if self.auto_display:
            figure.canvas.draw()
            figure.canvas.flush_events()

        return figure, ax

    def draw_environment(
        self,
        environment: Environment,
        aircraft_color: str | tuple[float, float, float, float] = (0.31, 0.31, 0.31, 1.0),
        actions_log: list[Action] | None = None,
    ):
        """
        Draw the Environment.

        Parameters
        ----------
        environment: Environment
            The Environment to draw
        aircraft_color: str
            Required parameter, named colour (str) i.e. red, green, blue, black. Or rgba colour defined as a tuple.
            Default = black
        actions_log: list[Action] | None
            Optional parameter, the log of actions
            Default = None
        """

        # first, clean up traces_dict: for aircraft pop blips for
        # aircraft that is no more in the environment.
        s1 = set(self.traces_dict.keys())
        s2 = set(environment.aircraft.keys())
        to_clean_up = s1.difference(s2)
        for name in to_clean_up:
            del self.traces_dict[name]

        self.draw_environment_time(environment)

        self.draw_airspace(environment.airspace)

        if self.render_routes:
            for aircraft in environment.aircraft.values():
                if aircraft.flight_plan is None:
                    warnings.warn(
                        UserWarning(f"{aircraft.callsign} has flight_plan set to None."),
                        stacklevel=2,
                    )
                    continue

                s1 = set(aircraft.flight_plan.route.filed)
                s2 = set(environment.airspace.fixes.places.keys())
                if len(s1.difference(s2)) > 0:
                    warnings.warn(
                        UserWarning(f"{aircraft.callsign} fix flight_plan does not exist in the airspace fixes."),
                        stacklevel=2,
                    )
                    continue

                if len(aircraft.flight_plan.route.filed) < 2:
                    warnings.warn(
                        UserWarning(f"{aircraft.callsign} has only one fix in flight_plan."),
                        stacklevel=2,
                    )
                    continue

                self.draw_route(aircraft.flight_plan.route, environment.airspace.fixes)

        self.draw_all_aircraft(environment, aircraft_color)

        if self.display_actions and actions_log is not None and len(actions_log) > 0:
            self.add_action_to_queue(actions_log, environment)

        if self.display_actions:
            self.draw_action_queue()

    def draw_environment_time(self, environment: Environment):
        """
        Add Environment time and step count to visualisation.

        Parameters
        ----------
        environment: Environment
            Required parameter, environment.
        """

        figure = self.get_figure()
        ax = figure.get_axes()[0]

        time_label = time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime(environment.time))
        ax.text(0.98, 0.02, time_label, horizontalalignment="right", transform=ax.transAxes)
        step_label = f"Step: {self.counter:5d}"
        self.counter += 1
        ax.text(7 / 8, 0.97, step_label, transform=ax.transAxes)
        return

    def draw_airspace(self, airspace: Airspace, include_fixes: bool = True):
        """
        Draw an airspace.
        """

        for sector in airspace.sectors.values():
            self.draw_sector(sector)

        if include_fixes:
            self.draw_fixes(airspace.fixes)

    def draw_sector(self, sector: Sector):
        """
        Draw a sector.
        """

        for volume in sector.volumes:
            self.draw_area(volume.area)

        for volume in sector.volumes:
            self.draw_volume(volume)

    def draw_volume(self, volume: Volume):
        """
        Draw a volume.
        """

        # 0 => longitude, 1 => latitude
        points = [self.lonlat_to_xy(coord[0], coord[1]) for coord in volume.area.boundary.exterior.coords]
        xy = np.array(points, dtype=np.float32)

        f = max(min((volume.min_fl + volume.max_fl) / 800.0, 1.0), 0.0)
        fill = (0.0, (1 - f), f, 0.1)  # rgba

        ax = self.get_figure().get_axes()[0]
        ax.fill(xy[:, 0], xy[:, 1], color=fill)

    def draw_area(self, area: Area):
        """
        Draw an area.
        """

        # 0 => longitude, 1 => latitude
        points = [self.lonlat_to_xy(coord[0], coord[1]) for coord in area.boundary.exterior.coords]
        xy = np.array(points, dtype=np.float32)

        ax = self.get_figure().get_axes()[0]
        ax.fill(xy[:, 0], xy[:, 1], color="#FAFAFA")

    def draw_fixes(self, fixes: Fixes):
        """
        Draw a set of fixes.
        """

        if self.render_fixes_list:
            for name in self.render_fixes_list:
                if not self.is_out_of_plot_area(fixes.places[name]):
                    self.draw_pos2d(fixes.places[name], name)
        else:
            if self.render_fixes:
                if self.render_sys_fixes:
                    for name, location in fixes.places.items():
                        if not self.is_out_of_plot_area(location):
                            self.draw_pos2d(location, name)
                else:
                    for name, location in fixes.places.items():
                        if bool(re.search(r"\d+", name)) is False and self.is_out_of_plot_area(location) is False:
                            self.draw_pos2d(location, name)

    def draw_route(self, route: Route, fixes: Fixes):
        """
        Draw a route.
        """

        ax = self.get_figure().get_axes()[0]

        for location_a, location_b in zip(route.filed, route.filed[1:], strict=False):
            start = fixes.places[location_a]
            end = fixes.places[location_b]

            start_x, start_y = self.lonlat_to_xy(start.lon, start.lat)
            end_x, end_y = self.lonlat_to_xy(end.lon, end.lat)

            xdata = np.array([start_x, end_x], dtype=np.float32)
            ydata = np.array([start_y, end_y], dtype=np.float32)
            ax.plot(xdata, ydata, color="#DDDDDD", linewidth=max(self.scale * 1e-3, 1.5))

    def draw_traces(
        self,
        aircraft: Aircraft,
        blip_color: str | tuple[float, float, float, float],
        environment: Environment | None = None,
    ):
        """
        Draw the last blips positions of the aircraft based on timestep as grey circles
        """

        ax = self.get_figure().get_axes()[0]

        if self.is_out_of_plot_area(aircraft.pos2d()):
            return

        name = aircraft.callsign

        # ensure the plot colour is in rgba format
        if not isinstance(blip_color, tuple):  # if not rgba
            blip_color = to_rgba(blip_color)

        if environment is not None and self.is_background(aircraft, environment):
            # reduce the alpha for background aircraft
            _blip_color = list(blip_color)
            _blip_color[-1] = max(0.2, _blip_color[-1] - 0.7)
            _blip_color = tuple(_blip_color)
        else:
            _blip_color = blip_color

        # now update the blips for aircraft that are active in the environment.
        # update the dictionary of radar trace positions with current
        # aircraft position or clear it stepwise if aircraft has left
        # the environment
        if name not in self.traces_dict:
            self.traces_dict[name] = deque(maxlen=self.blips)

        self.traces_dict[name].append(Pos2D(aircraft.lat, aircraft.lon))

        # draw the blips
        size = int(0.8 * DEFAULT_MARKER_SIZE) ** 2

        for i in range(len(self.traces_dict[name]) - 1):
            blip = self.traces_dict[name][i]
            x, y = self.lonlat_to_xy(blip.lon, blip.lat)
            # a high zorder ensure that the plot is drawn on top of other drawing (at the same location)
            # with a lower zorder.
            ax.scatter(x, y, s=size, color=_blip_color, linewidths=self.scale * 0.3e-4, zorder=1000)

    def draw_all_aircraft(self, environment: Environment, aircraft_color: str | tuple[float, float, float, float]):
        """
        Draw all aircraft in the environment.
        """

        for aircraft in environment.aircraft.values():
            self.draw_aircraft(aircraft, aircraft_color, environment)

    def draw_aircraft(
        self,
        aircraft: Aircraft,
        aircraft_color: str | tuple[float, float, float, float],
        environment: Environment | None = None,
    ):
        """
        Draw an aircraft.
        """

        ax = self.get_figure().get_axes()[0]

        # ensure the plot colour is in rgba format
        if not isinstance(aircraft_color, tuple):  # if not rgba
            aircraft_color = to_rgba(aircraft_color)

        name = aircraft.callsign
        if self.is_out_of_plot_area(aircraft.pos2d()):
            return

        # process the color argument
        if environment is not None and self.is_background(aircraft, environment):
            # reduce the alpha for background aircraft
            _aircraft_color = list(aircraft_color)
            _aircraft_color[-1] = max(0.1, _aircraft_color[-1] - 0.8)
            _aircraft_color = tuple(_aircraft_color)
        else:
            _aircraft_color = aircraft_color

        if environment is not None and self.sector_name is not None:
            exit_coordination = environment.coordinations.get(name, from_sector=self.sector_name)
        else:
            exit_coordination = []
        if aircraft.current_sector is not None and exit_coordination != []:
            exit_fl = exit_coordination[0].fl
        else:
            # Q: is this a good default or should we pick some other value?
            exit_fl = aircraft.selected_fl

        labels = [
            f"{name} {int(exit_fl)}",
            f"{int(aircraft.fl)} {int(aircraft.selected_fl)}",
        ]

        # first, draw the traces/blips
        if self.blips > 0:
            self.draw_traces(aircraft, aircraft_color, environment)

        # draw aircraft current position as a dot and an asterisk
        x, y = self.lonlat_to_xy(aircraft.lon, aircraft.lat)
        size = int(0.8 * DEFAULT_MARKER_SIZE) ** 2
        ## dot
        # a high zorder ensure that the plot is drawn on top of other drawing (at the same location)
        # with a lower zorder.
        ax.scatter(x, y, s=size, color=_aircraft_color, zorder=1000)

        ## asterisk
        for angle_delta in [0, 45, 90, -45]:
            angle = (aircraft.heading + angle_delta) % 360.0
            new = aircraft.pos2d().forward(1.0, angle).location  # format: (lat, lon)
            new_x, new_y = self.lonlat_to_xy(new[1], new[0])
            ax.plot([x, new_x], [y, new_y], color=_aircraft_color, zorder=1000)

            angle = (angle + 180.0) % 360.0
            new = aircraft.pos2d().forward(1.0, angle).location  # format: (lat, lon)
            new_x, new_y = self.lonlat_to_xy(new[1], new[0])
            ax.plot([x, new_x], [y, new_y], color=_aircraft_color, zorder=1000)

        if len(labels) > 0:
            txt = "\n  " + "\n  ".join(labels)
            ax.text(
                x,
                y,
                txt,
                fontsize=DEFAULT_FONT_SIZE * 0.8,
                fontstretch="ultra-condensed",
                color=_aircraft_color,
                verticalalignment="top",
            )

        if self.render_sep_bound:
            # Find the radius of the circle in display coordinates by moving the lateral separation
            # distance in latitude and longitude and converting.
            gh = GeoHelper()
            X = []
            Y = []
            for angle in np.linspace(0, 360, 36, endpoint=True):
                _lon, _lat = gh.forward(aircraft.lon, aircraft.lat, distance=self.aircraft_lateral_sep, heading=angle)
                x_r, y_r = self.lonlat_to_xy(_lon, _lat)
                X.append(x_r)
                Y.append(y_r)
            ax.plot(X, Y, color="chocolate", linewidth=0.5)

    def draw_plan(self, plan: Plan):
        """
        Draw a plan.
        """

        ax = self.get_figure().get_axes()[0]

        start = plan.positions[0]
        start_x, start_y = self.lonlat_to_xy(start.lon, start.lat)
        for end in plan.positions[1:]:
            end_x, end_y = self.lonlat_to_xy(end.lon, end.lat)
            ax.plot(
                [start_x, end_x],
                [start_y, end_y],
                linewidth=max(self.scale * 1e-3, 1.5),
                marker="o",
                color=DEFAULT_PLAN_COLOR,
            )
            ax.annotate(
                f"{int(start.fl)}",
                (start_x, start_y),
                textcoords="data",
                fontsize=(DEFAULT_FONT_SIZE * 0.7),
                fontstretch="extra-condensed",
                zorder=1200,  # this ensures the text is drawn above the black and gold scatter plot below.
            )
            start_x, start_y = end_x, end_y

        ax.annotate(
            f"{int(plan.positions[-1].fl)}",
            (end_x, end_y),
            textcoords="data",
            fontsize=(DEFAULT_FONT_SIZE * 0.7),
            fontstretch="extra-condensed",
            zorder=1200,  # this ensures the text is drawn above the black and gold scatter plot below.
        )

        # a high zorder ensure that the plot is drawn on top of other drawing (at the same location)
        # with a lower zorder.
        x, y = self.lonlat_to_xy(plan.positions[0].lon, plan.positions[0].lat)
        ax.scatter(x, y, color="black", marker="o", zorder=1000)
        x, y = self.lonlat_to_xy(plan.positions[-1].lon, plan.positions[-1].lat)
        ax.scatter(x, y, color="gold", marker="o", zorder=1000)

    def draw_trajectory(self, trajectory: list[Pos4D]):
        """
        Draw a trajectory.
        """

        for control_point in trajectory:
            self.draw_pos4d(control_point)

    def draw_pos2d(self, location: Pos2D, label: str = ""):
        """
        Draw a location.
        Parameters
        ----------
        location: Pos2D
            Required parameter, the position (longitude and latitude) to draw the item
        label: str
            Required parameter, text label present
        """

        ax = self.get_figure().get_axes()[0]

        edge_size = 1.5 + (self.scale * 2e-4)  # 1.5 is matplotlib's default
        x, y = self.lonlat_to_xy(location.lon, location.lat)
        ax.scatter(x, y, color="chocolate", linewidths=edge_size)

        if len(label) > 0:
            ax.text(x, y, " " + label, fontsize=DEFAULT_FONT_SIZE * 0.8)

    def draw_pos3d(self, position: Pos3D, label: str = ""):
        """
        Draw a position.
        """

        ax = self.get_figure().get_axes()[0]

        f = max(min(position.fl / 400.0, 1.0), 0.0)
        red = 0.0
        green = 1.0 - f
        blue = f

        fill_color = (red, green, blue, 1.0)
        edge_size = 1.5 + (self.scale * 2e-4)  # 1.5 is matplotlib's default
        x, y = self.lonlat_to_xy(position.lon, position.lat)
        ax.scatter(x, y, color=fill_color, linewidths=edge_size, edgecolors="black")

        if len(label) > 0:
            ax.text(x, y, " " + label, fontsize=DEFAULT_FONT_SIZE * 0.8)

    def draw_pos4d(self, control_point: Pos4D, label: str = ""):
        """
        Draw a control point.
        """

        ax = self.get_figure().get_axes()[0]

        f = max(min(control_point.fl / 400.0, 1.0), 0.0)
        r = 0.0
        g = 1.0 - f
        b = f

        inner_fill_color = (r, g, b)
        x, y = self.lonlat_to_xy(control_point.lon, control_point.lat)
        ax.scatter(x, y, color=inner_fill_color, linewidths=0.0)

        # t = np.fmod(control_point.time, 900.0) / 900.0
        # outer_fill = (t, 0.7, 0.4)  # hsv format
        # outer_fill = colorsys.hsv_to_rgb(*outer_fill)  # now rgb format
        # edge_size = 1.5 + (self.scale * 2e-4)  # 1.5 is matplotlib's default
        # ax.scatter(x, y, color=inner_fill_color, linewidths=edge_size, edgecolors=outer_fill)

        if len(label) > 0:
            ax.text(x, y, " " + label, fontsize=DEFAULT_FONT_SIZE * 0.8)

    def add_action_to_queue(self, actions: list[Action], environment: Environment):
        """
        Displays a list of the most recent actions and the time they were taken
        Parameters
        ----------
        actions: list(Action)
            Required parameter, list of actions at given step
        environment: Environment
            Required parameter, environment
        """

        for act in actions:
            qstring = time.strftime("%H:%M:%S", time.gmtime(environment.time)) + " " + str(act)
            self.action_queue.append(qstring)

    def draw_action_queue(self):
        """
        Displays a list of the most recent actions and the time they were taken
        """
        ax = self.get_figure().get_axes()[0]

        # if there are empty and non-empty slots in the action_queue, ensure that the
        # non-empty are moved to the earlier indexes.
        actions = [action_str for action_str in self.action_queue if len(action_str) > 0]

        # reverse order (the most recent actions are displayed at the top)
        actions = actions[::-1]

        for _ in range(self.action_queue.maxlen - len(actions)):
            actions.append("")

        action_string = "\n".join(actions)
        action_string = "Actions\n" + action_string
        # ax.text(0.9, 0.5, action_string,
        ax.text(
            0.2,
            -0.2,
            action_string,
            transform=ax.transAxes,
            fontsize=DEFAULT_FONT_SIZE,
            fontstretch="extra-condensed",
            verticalalignment="center",
        )

    def is_out_of_plot_area(self, location: Pos2D) -> bool:
        """
        Checks whether or not a location is outside the defined plot area.

        Parameters
        ----------
        location: Pos2D
            Required parameter, the location/point to check.

        Returns
        -------
        bool :
            the status of the location. True if it is outside the defined plot area, False otherwise.
        """

        x, y = self.lonlat_to_xy(location.lon, location.lat)
        # the bool casting ensures that a numpy bool is not returned
        return bool(x < self.min_x or x > self.max_x or y < self.min_y or y > self.max_y)

    def is_background(self, aircraft: Aircraft, environment: Environment) -> bool:
        """
        Checks whether or not an aircraft is a background traffic.

        Parameters
        ----------
        aircraft: Aircraft
            Required parameter, the aircraft to check
        environment: Environment
            Required parameter, environment

        Returns
        -------
        bool :
            the status of the aircraft. True if background, False otherwise.
        """

        if self.sector_name is not None:
            entry_coordination = environment.coordinations.get(aircraft.callsign, to_sector=self.sector_name)

            # if the aircraft will not enter the active sector being displayed
            # in the radar, then set it as a background traffic (i.e., `True`).
            background_status = len(entry_coordination) == 0

        else:
            # no sector was set in the radar class.
            # assumption: the aircraft is not a background traffic.
            background_status = False

        return background_status
