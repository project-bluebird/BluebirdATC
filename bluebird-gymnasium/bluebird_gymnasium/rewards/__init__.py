from bluebird_gymnasium.utils.module_registry import ModuleRegistry

registry_reward_fn = ModuleRegistry()
base_pkg = "bluebird_gymnasium.rewards"

# action penalty
mod_name = f"{base_pkg}.action_penalty"
registry_reward_fn.register("action_penalty_memory", f"{mod_name}:action_penalty_memory")
registry_reward_fn.register("action_penalty_const", f"{mod_name}:action_penalty_const")
registry_reward_fn.register("action_penalty_thresh", f"{mod_name}:action_penalty_thresh")

# expeditious
mod_name = f"{base_pkg}.expeditious"
registry_reward_fn.register("expeditious_const", f"{mod_name}:expeditious_const")
registry_reward_fn.register("expeditious_linear", f"{mod_name}:expeditious_linear")
registry_reward_fn.register("expeditious_quad", f"{mod_name}:expeditious_quad")
registry_reward_fn.register("expeditious_exp", f"{mod_name}:expeditious_exp")

# lateral centreline distance
mod_name = f"{base_pkg}.lateral_centreline_distance"
registry_reward_fn.register(
    "lateral_centreline_distance_exp",
    f"{mod_name}:lateral_centreline_distance_exp",
)
registry_reward_fn.register(
    "lateral_centreline_distance_linear",
    f"{mod_name}:lateral_centreline_distance_linear",
)
registry_reward_fn.register(
    "lateral_centreline_distance_special",
    f"{mod_name}:lateral_centreline_distance_special",
)
registry_reward_fn.register(
    "lateral_centreline_distance_quad",
    f"{mod_name}:lateral_centreline_distance_quad",
)
registry_reward_fn.register(
    "lateral_centreline_distance_shaped",
    f"{mod_name}:lateral_centreline_distance_shaped",
)

# lateral next fix proximity
mod_name = f"{base_pkg}.lateral_next_fix_proximity"
registry_reward_fn.register(
    "lateral_next_fix_proximity_dist_exp",
    f"{mod_name}:lateral_next_fix_proximity_dist_exp",
)
registry_reward_fn.register(
    "lateral_next_fix_proximity_bacnf",
    f"{mod_name}:lateral_next_fix_proximity_bacnf",
)
registry_reward_fn.register(
    "lateral_next_fix_proximity_dist_bpfnf",
    f"{mod_name}:lateral_next_fix_proximity_bpfnf",
)

# lateral termination check
mod_name = f"{base_pkg}.lateral_termination_check"
registry_reward_fn.register("lateral_termination_check_mac", f"{mod_name}:lateral_termination_check_mac")
registry_reward_fn.register("lateral_termination_check_sac", f"{mod_name}:lateral_termination_check_sac")

# position
mod_name = f"{base_pkg}.position"
registry_reward_fn.register("position_status_const", f"{mod_name}:position_status_const")

# safety
mod_name = f"{base_pkg}.safety"
registry_reward_fn.register("safety_simple_avoidance_exp", f"{mod_name}:safety_simple_avoidance_exp")
registry_reward_fn.register("safety_simple_avoidance_nvl", f"{mod_name}:safety_simple_avoidance_nvl")

# climb descent
mod_name = f"{base_pkg}.climb_descent"
registry_reward_fn.register("climb_target_exp", f"{mod_name}:climb_target_exp")
registry_reward_fn.register("climb_target_linear", f"{mod_name}:climb_target_linear")
registry_reward_fn.register("climb_target_quad", f"{mod_name}:climb_target_quad")
registry_reward_fn.register("descent_target_exp", f"{mod_name}:descent_target_exp")
registry_reward_fn.register("descent_target_linear", f"{mod_name}:descent_target_linear")
registry_reward_fn.register("descent_target_quad", f"{mod_name}:descent_target_quad")

registry_reward_fn.register("overflier_const", f"{mod_name}:overflier_const")
registry_reward_fn.register("overflier_exp", f"{mod_name}:overflier_exp")
registry_reward_fn.register("overflier_linear", f"{mod_name}:overflier_linear")
registry_reward_fn.register("overflier_quad", f"{mod_name}:overflier_quad")

# conflict resolution
mod_name = f"{base_pkg}.conflict_resolution"
registry_reward_fn.register("conflict_resolution_exp", f"{mod_name}:conflict_resolution_exp")
registry_reward_fn.register("conflict_resolution_tanh", f"{mod_name}:conflict_resolution_tanh")

# fly parallel to route
mod_name = f"{base_pkg}.route_parallel"
registry_reward_fn.register("route_parallel_const", f"{mod_name}:route_parallel_const")
registry_reward_fn.register("route_parallel_linear", f"{mod_name}:route_parallel_linear")
registry_reward_fn.register("route_parallel_quad", f"{mod_name}:route_parallel_quad")
registry_reward_fn.register("route_parallel_exp", f"{mod_name}:route_parallel_exp")

# custom rewards
mod_name = f"{base_pkg}.custom.reward_drlan"
registry_reward_fn.register("reward_drlan", f"{mod_name}:reward_drlan")
registry_reward_fn.register("custom_reward", f"{mod_name}:custom_reward_fn")
mod_name = f"{base_pkg}.custom.custom_reward"
registry_reward_fn.register(
    "route_progress_terminal_reward",
    f"{mod_name}:route_progress_terminal_reward",
)
registry_reward_fn.register(
    "lateral_termination_check_sac_env",
    f"{mod_name}:lateral_termination_check_sac_env",
)
registry_reward_fn.register(
    "lateral_termination_check_mac_env",
    f"{mod_name}:lateral_termination_check_mac_env",
)
registry_reward_fn.register(
    "anti_loiter_route_rejoin_reward",
    f"{mod_name}:anti_loiter_route_rejoin_reward",
)
