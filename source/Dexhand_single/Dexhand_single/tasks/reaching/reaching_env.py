import torch
from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    quat_mul,
    quat_inv,
    quat_apply,
    quat_error_magnitude,
    create_rotation_matrix_from_view,
    quat_from_matrix,
    axis_angle_from_quat,
    quat_from_angle_axis,
)
from .reaching_env_cfg import ReachingEnvCfg


class ReachingEnv(DirectRLEnv):
    cfg: ReachingEnvCfg

    def __init__(self, cfg: ReachingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.prev_dist_to_target = torch.zeros(self.num_envs, device=self.device)
        self.prev_ang_dist_to_target = torch.zeros(self.num_envs, device=self.device)
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_quat = torch.zeros((self.num_envs, 4), device=self.device)

        self.ball_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        self.ball_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.ball_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.ball = RigidObject(self.cfg.ball)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["ball"] = self.ball
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        current_pos = self.robot.data.root_pos_w
        current_quat = self.robot.data.root_quat_w

        # The agent's actions are in [-1, 1] range, map them to [-max_vel, +max_vel]
        target_lin_vel_local = self.actions[:, :3] * self.cfg.max_linear_velocity

        # For angular velocity, the policy outputs a 4D quaternion-like action.
        # We will normalize it to get a unit quaternion, convert to axis-angle,
        # and then scale the axis-angle vector by max_angular_velocity.
        delta_orn_quat_action = torch.nn.functional.normalize(
            self.actions[:, 3:], p=2, dim=1
        )

        # Convert action quaternion to rotation vector (axis * angle)
        action_rotation_vector = axis_angle_from_quat(delta_orn_quat_action)

        # Scale the rotation vector by max_angular_velocity to get desired angular velocity
        target_ang_vel = action_rotation_vector * self.cfg.max_angular_velocity

        # Transform local linear velocity to world frame
        world_frame_lin_vel = quat_apply(current_quat, target_lin_vel_local)

        # Clamp velocities to their maximum magnitudes
        clamped_lin_vel = torch.clamp(
            world_frame_lin_vel,
            min=-self.cfg.max_linear_velocity,
            max=self.cfg.max_linear_velocity,
        )
        clamped_ang_vel = torch.clamp(
            target_ang_vel,
            min=-self.cfg.max_angular_velocity,
            max=self.cfg.max_angular_velocity,
        )

        # Compute the final pose based on the clamped velocities
        final_pos = current_pos + clamped_lin_vel * self.physics_dt

        clamped_angle = torch.norm(clamped_ang_vel, dim=-1) * self.physics_dt
        clamped_axis = clamped_ang_vel / (
            torch.norm(clamped_ang_vel, dim=-1, keepdim=True) + 1e-6
        )
        clamped_delta_q = quat_from_angle_axis(clamped_angle, clamped_axis)
        final_quat = quat_mul(clamped_delta_q, current_quat)

        # Write the clamped state to the simulation
        root_state = torch.cat(
            [final_pos, final_quat, clamped_lin_vel, clamped_ang_vel], dim=-1
        )
        self.robot.write_root_state_to_sim(root_state)

        # Set finger joints to default position
        self.robot.set_joint_position_target(self.robot.data.default_joint_pos)

    def _get_observations(self) -> dict:
        robot_lin_vel = self.robot.data.root_lin_vel_w
        robot_ang_vel = self.robot.data.root_ang_vel_w
        robot_pos = self.robot.data.root_pos_w
        robot_quat = self.robot.data.root_quat_w

        target_pos = self.target_pos
        target_quat = self.target_quat

        # Calculate relative
        relative_pos, relative_quat = get_relative_pose(
            robot_quat, robot_pos, target_quat, target_pos
        )

        obs = torch.cat(
            [
                robot_lin_vel,
                robot_ang_vel,
                relative_pos,
                relative_quat,
            ],
            dim=-1,
        )
        observations = {"policy": obs, "critic": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        robot_pos = self.robot.data.root_pos_w
        robot_quat = self.robot.data.root_quat_w
        target_pos = self.target_pos
        target_quat = self.target_quat

        # Calculate current distances
        current_dist_to_target = torch.norm(robot_pos - target_pos, dim=-1)
        current_ang_dist_to_target = quat_error_magnitude(robot_quat, target_quat)

        # Check for success
        terminated_success = (current_dist_to_target < self.cfg.pos_tolerance) & (
            current_ang_dist_to_target < self.cfg.orn_tolerance
        )

        # Compute the reward
        reward = compute_rewards(
            current_dist_to_target,
            self.prev_dist_to_target,
            current_ang_dist_to_target,
            self.prev_ang_dist_to_target,
            self.cfg.rew_scale_pos_potential,
            self.cfg.rew_scale_orn_potential,
            self.cfg.rew_success_bonus,
            terminated_success,
        )

        # Update the previous distance buffers
        self.prev_dist_to_target = current_dist_to_target
        self.prev_ang_dist_to_target = current_ang_dist_to_target

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        robot_pos = self.robot.data.root_pos_w
        robot_quat = self.robot.data.root_quat_w
        target_pos = self.target_pos
        target_quat = self.target_quat

        # Calculate current distances
        current_dist_to_target = torch.norm(robot_pos - target_pos, dim=-1)
        current_ang_dist_to_target = quat_error_magnitude(robot_quat, target_quat)

        # Check for success
        terminated_success = (current_dist_to_target < self.cfg.pos_tolerance) & (
            current_ang_dist_to_target < self.cfg.orn_tolerance
        )

        return terminated_success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        num_resets = len(env_ids)

        # Reset robot root state to fixed initial pose
        root_state = self.robot.data.default_root_state[env_ids]
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        robot_root_pos_w = self.robot.data.root_pos_w[env_ids]

        # Generate random offsets for the object position
        min_object_offset = torch.tensor(
            self.cfg.object_spawn_pos_min, device=self.device
        )
        max_object_offset = torch.tensor(
            self.cfg.object_spawn_pos_max, device=self.device
        )
        object_offset = min_object_offset + (
            max_object_offset - min_object_offset
        ) * torch.rand(num_resets, 3, device=self.device)
        # The object_pos is relative to the robot's initial position
        object_pos = robot_root_pos_w + object_offset

        up_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            num_resets, 1
        )

        forward_vec = torch.nn.functional.normalize(
            object_pos - robot_root_pos_w, p=2, dim=-1
        )
        right_vec = torch.nn.functional.normalize(
            torch.cross(up_axis, forward_vec, dim=-1)
        )
        up_vec = torch.cross(forward_vec, right_vec, dim=-1)

        rot_matrix_3x3 = torch.stack([right_vec, up_vec, forward_vec], dim=-1)

        self.target_quat[env_ids] = quat_from_matrix(rot_matrix_3x3)

        grasp_offset = torch.tensor(self.cfg.grasp_offset, device=self.device)

        rotated_grasp_offset = quat_apply(
            self.target_quat[env_ids], -grasp_offset.expand(num_resets, -1)
        )
        self.target_pos[env_ids] = object_pos + rotated_grasp_offset

        ball_pos = object_pos

        # Reset finger joint positions
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Set ball position
        ball_state = torch.cat(
            [
                ball_pos,
                self.ball_quat[env_ids],
                self.ball_lin_vel[env_ids],
                self.ball_ang_vel[env_ids],
            ],
            dim=-1,
        )
        self.ball.write_root_state_to_sim(ball_state, env_ids=env_ids)

        # Store the initial distance and angular distance to the target
        robot_pos = self.robot.data.root_pos_w[env_ids]
        robot_quat = self.robot.data.root_quat_w[env_ids]
        target_pos = self.target_pos[env_ids]
        target_quat = self.target_quat[env_ids]

        self.prev_dist_to_target[env_ids] = torch.norm(robot_pos - target_pos, dim=-1)
        self.prev_ang_dist_to_target[env_ids] = quat_error_magnitude(
            robot_quat, target_quat
        )

        super()._reset_idx(env_ids)


@torch.jit.script
def get_relative_pose(robot_quat, robot_pos, target_quat, target_pos):
    robot_quat_inv = quat_inv(robot_quat)
    relative_quat = quat_mul(robot_quat_inv, target_quat)
    world_vec = target_pos - robot_pos
    relative_pos = quat_apply(robot_quat_inv, world_vec)
    return relative_pos, relative_quat


@torch.jit.script
def compute_rewards(
    current_dist: torch.Tensor,
    prev_dist: torch.Tensor,
    current_ang_dist: torch.Tensor,
    prev_ang_dist: torch.Tensor,
    rew_pos_potential_scale: float,
    rew_orn_potential_scale: float,
    rew_success_bonus: float,
    terminated: torch.Tensor,
) -> torch.Tensor:
    pos_reward = rew_pos_potential_scale * (prev_dist - current_dist)
    orn_reward = rew_orn_potential_scale * (prev_ang_dist - current_ang_dist)
    success_reward = rew_success_bonus * terminated.float()
    reward = pos_reward + orn_reward + success_reward

    return reward
