# Dexhand Reaching Task: Plan and Requirements (Updated)

* **Robot Configuration (Floating Hand):**
  * The Dexhand will be treated as a floating, kinematic body.
  * The policy will directly control the 6-DoF pose (position and orientation) of the hand's base.
  * For this initial task, the hand's fingers will be kept in a static, neutral pose and will not be controlled by the policy.

* **Target Pose:**
  * The target is a fixed numerical pose (position and orientation) defined in the environment configuration.
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
