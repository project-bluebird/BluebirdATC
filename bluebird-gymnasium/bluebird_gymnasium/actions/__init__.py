# constants
# placed at the top to avoid circular import issues
ACTION_NOOP = 0
NUM_NOOP_ACTIONS = 1
DEFAULT_RELATIVE_HEADING = 10
DEFAULT_RELATIVE_CLIMB_DESCENT = 10
DEFAULT_RELATIVE_SPEED = 10
DEFAULT_ROUTE_DIRECT = 1
DEFAULT_ROUTE_PARALLEL = 1

DEFAULT_INTERVAL_FL = 10
DEFAULT_INTERVAL_HEADING = 5

from bluebird_gymnasium.utils.module_registry import ModuleRegistry  # noqa: E402

registry_actions = ModuleRegistry()
base_pkg = "bluebird_gymnasium.actions"

# simple: heading
mod_name = f"{base_pkg}.simple.heading"
registry_actions.register("simple_heading_left", f"{mod_name}:heading_left")
registry_actions.register("simple_heading_right", f"{mod_name}:heading_right")
registry_actions.register(
    "simple_heading_route_parallel", f"{mod_name}:heading_route_parallel"
)
registry_actions.register(
    "simple_heading_maintain_current", f"{mod_name}:heading_maintain_current"
)

# simple: climb/descent
mod_name = f"{base_pkg}.simple.climb_descent"
registry_actions.register("simple_fl_climb", f"{mod_name}:fl_climb")
registry_actions.register("simple_fl_descent", f"{mod_name}:fl_descent")
registry_actions.register(
    "simple_fl_intermediate", f"{mod_name}:fl_intermediate"
)
registry_actions.register("simple_fl_exit", f"{mod_name}:fl_exit")

# simple: speed
mod_name = f"{base_pkg}.simple.speed"
registry_actions.register("simple_speed_increase", f"{mod_name}:speed_increase")
registry_actions.register("simple_speed_decrease", f"{mod_name}:speed_decrease")
registry_actions.register(
    "simple_speed_maintain_current", f"{mod_name}:speed_maintain_current"
)
registry_actions.register(
    "simple_speed_choose_own", f"{mod_name}:speed_choose_own"
)

# simple: route direct
mod_name = f"{base_pkg}.simple.route_direct"
registry_actions.register("simple_route_direct", f"{mod_name}:route_direct")

# simple: outcomm
mod_name = f"{base_pkg}.simple.outcomm"
registry_actions.register("simple_outcomm", f"{mod_name}:outcomm")

# compound: heading, fly parallel
mod_name = f"{base_pkg}.compound.fly_parallel"
registry_actions.register(
    "compound_heading_fly_parallel", f"{mod_name}:compound_heading_fly_parallel"
)
