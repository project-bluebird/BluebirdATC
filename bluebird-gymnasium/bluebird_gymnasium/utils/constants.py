import os
from pathlib import Path

from bluebird_dt.utility.paths import LOG_DIR as REPLAY_DIR

############ simulator specifics
SECTOR_BACKGROUND = "background"
SIMULATION_LOG_DIR = REPLAY_DIR


############ distance (unit: nautical miles)
# distance within which an external agent needs to issue an outcomm action
# to an aircraft without incurring any penalty.
DISTANCE_TO_EXIT_THRESHOLD = 15.0

# exit window width
EXIT_WINDOW_WIDTH_DEFAULT = 5.0

# when sector is exited correctly: maximum distance away from the exit
# fix/position/window after which an aircraft stops being tracked.
DISTANCE_AWAY_FROM_EXIT_THRESHOLD = 25.0

# when sector is exited incorrectly: maximum distance away from the exit
# fix/position/window after which an aircraft stops being tracked.
DISTANCE_AWAY_FROM_INCORRECT_EXIT_THRESHOLD = 20.0

# for aircraft yet to incomm (enter) a sector. if its distance to the entry
# fix/position is greater than the threshold, ignore it, else, track aircraft.
DISTANCE_TO_ENTRY_THRESHOLD = 10.0

# distance to exit flight level.
# distance before top of descent from which descent reward function starts
# punishing the agent for not descending. added to make a more conservative
# descend.
DIST_EXIT_LEVEL_THRESHOLD = 10.0


############ lateral speed (unit: nautical miles per hour aka knots)
# assumed maximum (true air) speed for any aircraft
MAX_SPEED_TAS = 450


############ vertical speed (unit: feet per minute)
DEFAULT_RATE_OF_CLIMB_DESCENT = 750

############ rollout (future) trajectory
FUTURE_TRAJ_DURATION = 300  # unit: seconds
FUTURE_TRAJ_DIST = 50  # unit: nautical miles (NMI)


############ others
DEFAULT_RENDER_DIR = os.path.join(Path.home(), "bluebird_gymnasium_render/")
DUMMY_CALLSIGN_PREFIX = "DUMMY"
AC_ENTRY_STEPS = 50
STEPS_SINCE_ACTION_MAX = 10


############ custom fix positions
CUSTOM_FIX_PREFIX = "CUSTOM_FIX"
CUSTOM_FIX_BEFORE_X = CUSTOM_FIX_PREFIX + "///BEFORE///{0}"
CUSTOM_FIX_AFTER_X = CUSTOM_FIX_PREFIX + "///AFTER///{0}"
CUSTOM_FIX_BETWEEN_X_AND_Y = CUSTOM_FIX_PREFIX + "///BETWEEN///{0}///{1}"
CUSTOM_FIX_AT_X = CUSTOM_FIX_PREFIX + "__AT__{0}"

CUSTOM_FIX_CURRENT_POS = "CURRENT_POS"
CUSTOM_FIX_FUTURE_POS = "FUTURE_POS"
