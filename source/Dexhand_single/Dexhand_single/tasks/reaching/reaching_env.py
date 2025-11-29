import torch
from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_mul, quat_inv, quat_apply, quat_error_magnitude
from .reaching_env_cfg import ReachingEnvCfg


class ReachingEnv(DirectRLEnv):
    cfg: ReachingEnvCfg

    def __init__(self, cfg: ReachingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.prev_dist_to_target = torch.zeros(self.num_envs, device=self.device)
        self.prev_ang_dist_to_target = torch.zeros(self.num_envs, device=self.device)
        self.target_pos = torch.tensor(
            self.cfg.target_position, device=self.device
        ).repeat(self.num_envs, 1)
        self.target_quat = torch.tensor(
            self.cfg.target_orientation, device=self.device
        ).repeat(self.num_envs, 1)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        current_pos = self.robot.data.root_pos_w
        current_quat = self.robot.data.root_quat_w

        # Extract delta actions from policy output
        delta_pos_action = self.actions[:, :3] * self.cfg.action_scale_pos
        delta_orn_action = self.actions[:, 3:]

        # Normalize the delta orientation action to ensure it's a unit quaternion
        delta_orn_quat = torch.nn.functional.normalize(delta_orn_action, p=2, dim=1)

        new_pos = current_pos + delta_pos_action

        # Apply delta rotation
        new_quat = quat_mul(delta_orn_quat, current_quat)

        # Scale down the rotation by interpolating
        w = self.cfg.action_scale_rot
        final_quat = (1 - w) * current_quat + w * new_quat
        final_quat = torch.nn.functional.normalize(final_quat, p=2, dim=1)

        zero_velocities = torch.zeros_like(self.robot.data.root_vel_w)
        root_state = torch.cat([new_pos, final_quat, zero_velocities], dim=-1)
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
            self.actions,
            self.reset_buf,
            self.cfg.rew_scale_pos_potential,
            self.cfg.rew_scale_orn_potential,
            self.cfg.rew_success_bonus,
            self.cfg.action_penalty,
            terminated_success,
        )

        # Update the previous distance buffers
        self.prev_dist_to_target = current_dist_to_target
        self.prev_ang_dist_to_target = current_ang_dist_to_target

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Only terminate on success during training (when num_envs > 1)
        if self.num_envs == 1:
            terminated_success = torch.zeros_like(time_out)
        else:
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

        # Reset robot root state to fixed initial pose
        root_state = self.robot.data.default_root_state[env_ids]
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        # Reset finger joint positions
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

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
    actions: torch.Tensor,
    reset_buf: torch.Tensor,
    rew_pos_potential_scale: float,
    rew_orn_potential_scale: float,
    rew_success_bonus: float,
    action_penalty: float,
    terminated: torch.Tensor,
) -> torch.Tensor:
    pos_reward = rew_pos_potential_scale * (prev_dist - current_dist)
    orn_reward = rew_orn_potential_scale * (prev_ang_dist - current_ang_dist)
    success_reward = rew_success_bonus * terminated.float()
    action_cost = action_penalty * torch.sum(torch.square(actions), dim=1)
    reward = pos_reward + orn_reward + success_reward + action_cost

    return reward
