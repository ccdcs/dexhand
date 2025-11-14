# Dexhand Reaching Task: Plan and Requirements

This document outlines the requirements and implementation plan for the Dexhand `Reaching` reinforcement learning task.

## 1. Goal and Requirements

### Overall Goal
To develop a reinforcement learning policy that learns to control the Dexhand's wrist, moving it to a specified target object within the simulation.

### Detailed Requirements

*   **Robot Configuration (Floating Hand):**
    *   The Dexhand will be treated as a floating, kinematic body, meaning its movement is not governed by physics simulation forces like gravity or momentum.
    *   The policy will directly control the 6-DoF pose (position and orientation) of the hand's base.
    *   For this initial task, the hand's fingers will be kept in a static, neutral pose and will not be controlled by the policy.

*   **Target Object:**
    *   A simple sphere will be used as the target.
    *   For initial development, the target will be placed at a fixed, non-randomized position: **(0.5, 0.0, 0.5)**.
    *   The hand will start at a fixed, non-randomized position: **(0.0, 0.0, 0.7)**.

*   **Action Space:**
    *   A 6-dimensional continuous vector representing the **absolute target pose** for the hand's wrist: `(x, y, z, roll, pitch, yaw)`.

*   **Observation Space:**
    *   A state-based, continuous vector containing:
        *   Hand wrist position (3 values: x, y, z)
        *   Hand wrist orientation (4 values: quaternion w, x, y, z)
        *   Target object position (3 values: x, y, z)
        *   Current finger joint angles (6 values)

*   **Reward Function:**
    *   A primary reward that increases as the distance between the hand and the target decreases.
    *   A large bonus reward for successfully reaching the target.
    *   A small penalty applied to the magnitude of the actions to encourage smooth and efficient movements.

*   **Success and Termination:**
    *   An episode is considered a **success** if the hand's wrist moves within a **5cm radius** of the target object.
    *   An episode **terminates** if it reaches the success condition or if it exceeds the maximum time limit.

## 2. Detailed Implementation Plan

The implementation will be broken into two main phases: first configuring the environment, and second, implementing the environment's logic.

### Phase 1: Environment Configuration (`reaching_env_cfg.py`)
1.  **Import necessary modules:** Add imports for `SceneObjectCfg` and `SphereCfg`.
2.  **Define Target Object:** Create a `SceneObjectCfg` for a sphere with a radius of 0.05m. Set its `kinematic_enabled` property to `True` and give it a red color for visualization.
3.  **Re-configure Robot:**
    *   In `DOFBOT_CONFIG`, set `kinematic_enabled=True` and `disable_gravity=True` to make the hand a floating kinematic body.
    *   Remove the `actuators` dictionary to make the fingers passive.
    *   Update the initial `pos` to `(0.0, 0.0, 0.7)`.
4.  **Update Environment Settings in `ReachingEnvCfg`:**
    *   Set `action_space = 6` and `observation_space = 16`.
    *   Add the target configuration to the environment.
    *   Define new reward scales (`rew_scale_dist`, `rew_scale_success`) and a `success_tolerance` of `0.05`.
    *   Remove obsolete configuration parameters from the old task.

### Phase 2: Environment Logic (`reaching_env.py`)
1.  **Update Class Members:** Remove obsolete variables (e.g., `target_joint_pos`) and add new ones to store handles to the robot and target objects and their data.
2.  **Modify `_setup_scene()`:**
    *   Instantiate the robot and the target object using the configurations from `ReachingEnvCfg`.
    *   Add the robot and target to the scene's list of tracked assets.
3.  **Modify `_reset_idx()`:**
    *   Implement logic to reset the robot's root pose and finger joint positions to their default states at the beginning of each episode.
4.  **Modify `_get_observations()`:**
    *   Gather the robot's root position/orientation, the target's position, and the finger joint angles.
    *   Concatenate these values into a single observation tensor.
5.  **Modify `_apply_action()`:**
    *   Take the 6D action from the policy.
    *   Convert the 3D orientation part (roll, pitch, yaw) into a quaternion.
    *   Apply the resulting target pose (position + quaternion) to the robot's root using the appropriate Isaac Lab function (`write_root_pose_to_sim`).
6.  **Modify `_get_rewards()`:**
    *   Create a new `compute_rewards` function.
    *   It will calculate the distance between the hand and the target and compute the total reward based on the defined reward scales.
7.  **Modify `_get_dones()`:**
    *   Implement the termination logic. An episode will terminate upon success (distance < tolerance) or timeout.
