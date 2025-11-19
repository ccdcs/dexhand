# Dexhand Reaching Task: Plan and Requirements (Updated)

## 1. Goal and Requirements

### Overall Goal

To develop a reinforcement learning policy that learns to control the Dexhand's wrist, moving it to a specific target position and orientation within the simulation.

### Detailed Requirements

* **Robot Configuration (Floating Hand):**
  * The Dexhand will be treated as a floating, kinematic body.
  * The policy will directly control the 6-DoF pose (position and orientation) of the hand's base.
  * For this initial task, the hand's fingers will be kept in a static, neutral pose and will not be controlled by the policy.

* **Target Object:**
  * A simple sphere will be used as the target.
  * **Fixed Position:** (0.5, 0.0, 0.5)
  * **Fixed Orientation:** Identity quaternion (no rotation)

* **Robot Initial Pose:**
  * **Fixed Position:** (0.0, 0.0, 0.7)
  * **Fixed Orientation:** Identity quaternion (no rotation)

* **Action Space (7-dimensional, Continuous, Relative):**
  * `delta_position` (3 values: change in x, y, z)
  * `delta_orientation` (4 values: quaternion for change in orientation)
  * Actions are relative to the robot's current pose. The output quaternion will be normalized before being applied.

* **Observation Space (13-dimensional, Continuous):**
  * `robot_linear_velocity` (3 values: x, y, z)
  * `robot_angular_velocity` (3 values: x, y, z)
  * `relative_target_position` (3 values: x, y, z of target relative to robot's local frame)
  * `relative_target_orientation` (4 values: quaternion of target orientation relative to robot's local frame)

* **Reward Function (Simple, Potential-based):**
  * **`rew_scale_pos_potential = 10.0`**: Rewards decrease in positional distance.
  * **`rew_scale_orn_potential = 5.0`**: Rewards decrease in angular distance (orientation).
  * **`rew_success_bonus = 100.0`**: Large bonus for achieving both position and orientation goal.
  * **`action_penalty = -0.001`**: Small penalty on squared magnitude of actions.
  * **`timeout_penalty = -2.0`**: Penalty for episode termination due to timeout.

* **Success and Termination:**
  * An episode is considered a **success** if:
    * `current_distance < pos_tolerance` (e.05 meters)
    * AND `current_angular_distance < orn_tolerance` (e.1745 radians, ~10 degrees)
  * An episode **terminates** if it reaches the success condition or if it exceeds the maximum time limit.

## 2. Detailed Implementation Plan

### Phase 1: Environment Configuration (`reaching_env_cfg.py`)

1. **Define Target Object:** Configure `RigidObjectCfg` for a sphere with:
    * Fixed position: (0.5, 0.0, 0.5)
    * Fixed orientation: Identity quaternion
    * `kinematic_enabled=True`
2. **Re-configure Robot:**
    * Configure `ArticulationCfg` for the robot with:
    * Fixed initial position: (0.0, 0.0, 0.7)
    * Fixed initial orientation: Identity quaternion
    * `kinematic_enabled=True`, `disable_gravity=True`
    * Remove `actuators` to keep fingers passive (if they are present from a previous setup).
3. **Update Environment Settings in `ReachingEnvCfg`:**
    * Set `action_space = 7` and `observation_space = 13`.
    * Update `rew_scale_pos_potential`, `rew_scale_orn_potential`, `rew_success_bonus`, `action_penalty`, `timeout_penalty`.
    * Define `pos_tolerance = 0.05` and `orn_tolerance = 0.1745`.
    * Remove obsolete configuration parameters.

### Phase 2: Environment Logic (`reaching_env.py`)

1. **Helper Functions:** Implement (or import) helper functions for `get_angular_distance` and `get_relative_pose`.
    * `get_angular_distance(quat1, quat2)`: Calculates the angular difference between two quaternions.
    * `get_relative_pose(robot_quat, robot_pos, target_quat, target_pos)`: Calculates relative target position and orientation (as quaternion).

2. **Modify `_apply_action()`:**
    * Extract `delta_position` (3D) and `delta_orientation_quat` (4D) from `self.actions`.
    * Scale `delta_position` by `action_scale_pos`.
    * **Normalize `delta_orientation_quat` to ensure it's a unit quaternion.**
    * Calculate `new_position = current_robot_pos + delta_position`.
    * Calculate `new_orientation_quat = quat_mul(delta_orientation_quat, current_robot_quat)`.
    * Apply `new_position` and `new_orientation_quat` to the robot's root.
    * Set finger joints to their default static positions.

3. **Modify `_get_observations()`:**
    * Get `robot_linear_velocity`, `robot_angular_velocity`.
    * Get `robot_orientation_quat` (needed for relative calculations).
    * Get `robot_pos_w` (needed for relative calculations).
    * Get `target_pos_w`, `target_quat_w`.
    * Calculate `relative_target_pos_w`, `relative_target_quat_w` using `get_relative_pose` with `robot_orientation_quat` and `robot_pos_w`.
    * Concatenate all 13 dimensions into the observation tensor.

4. **Modify `_get_rewards()`:**
    * Calculate `current_distance = torch.norm(robot_pos - target_pos, dim=-1)`.
    * Calculate `current_angular_distance = get_angular_distance(robot_quat, target_quat)`.
    * Calculate `pos_potential_reward = rew_scale_pos_potential * (prev_dist - current_distance)`.
    * Calculate `orn_potential_reward = rew_scale_orn_potential * (prev_ang_dist - current_angular_distance)`.
    * Calculate `action_cost = action_penalty * torch.sum(torch.square(self.actions), dim=1)`.
    * `reward = pos_potential_reward + orn_potential_reward + action_cost`.
    * Apply `rew_success_bonus` if success condition met.
    * Update `prev_dist` and `prev_ang_dist` buffers.

5. **Modify `_get_dones()`:**
    * Calculate `success = (current_distance < pos_tolerance) & (current_angular_distance < orn_tolerance)`.
    * Return `terminated = success` and `time_out`.

6. **Modify `_reset_idx()`:**
    * Reset robot to its fixed initial pose (position and orientation).
    * Reset target to its fixed position and orientation.
    * Initialize `prev_dist` and `prev_ang_dist` buffers.

