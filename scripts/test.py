# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import torch  # 确保有torch导入

from isaaclab.app import AppLauncher
from compliance_utils import compliance_cl
# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
from scipy.optimize import fsolve

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

def compute_jacobian(joint_positions, R, D1, D2, L, S):
    """
    计算雅可比矩阵（根据论文公式）
    
    参数:
    joint_positions: 关节位置 [theta1, theta2, theta3, ...]
    R, D1, D2, L, S: 几何参数
    
    返回:
    jacobian: 雅可比矩阵 J = [∂θ1/∂l1  ∂θ1/∂l2]
                              [∂θ2/∂l1  ∂θ2/∂l2]
    """
    theta1, theta2 = joint_positions
    
    # 计算中间变量
    A = R * (1 - np.cos(theta2)) + D2 * np.sin(theta1)
    H = D1 + R * np.sin(theta2) + D2 * np.cos(theta2)
    V = L / 2
    
    # 计算 l1 和 l2 的表达式（从论文公式(1)）
    l1_squared = A**2 + (H * np.cos(theta1) - V * np.sin(theta1) - S)**2 + (H * np.sin(theta1) + V * np.cos(theta1) - V)**2
    l2_squared = A**2 + (H * np.cos(theta1) + V * np.sin(theta1) - S)**2 + (H * np.sin(theta1) - V * np.cos(theta1) + V)**2
    
    l1 = np.sqrt(l1_squared)
    l2 = np.sqrt(l2_squared)
    
    # ∂l1/∂θ1
    dl1_dtheta1 = (-H * np.sin(theta1) - V * np.cos(theta1)) * (H * np.cos(theta1) - V * np.sin(theta1) - S) / l1 + \
                  (H * np.cos(theta1) - V * np.sin(theta1)) * (H * np.sin(theta1) + V * np.cos(theta1) - V) / l1
    
    # ∂l1/∂θ2  
    dl1_dtheta2 = (R * np.sin(theta2) - D2 * np.sin(theta2)) * A / l1 + \
                  (R * np.cos(theta2) - D2 * np.sin(theta2)) * (H * np.cos(theta1) - V * np.sin(theta1) - S) / l1 + \
                  (R * np.cos(theta2) - D2 * np.cos(theta2)) * (H * np.sin(theta1) + V * np.cos(theta1) - V) / l1
    
    # ∂l2/∂θ1
    dl2_dtheta1 = (-H * np.sin(theta1) + V * np.cos(theta1)) * (H * np.cos(theta1) + V * np.sin(theta1) - S) / l2 + \
                  (H * np.cos(theta1) + V * np.sin(theta1)) * (H * np.sin(theta1) - V * np.cos(theta1) + V) / l2
    
    # ∂l2/∂θ2
    dl2_dtheta2 = (R * np.sin(theta2) - D2 * np.sin(theta2)) * A / l2 + \
                  (R * np.cos(theta2) - D2 * np.sin(theta2)) * (H * np.cos(theta1) + V * np.sin(theta1) - S) / l2 + \
                  (R * np.cos(theta2) - D2 * np.cos(theta2)) * (H * np.sin(theta1) - V * np.cos(theta1) + V) / l2
    
    J_forward = np.array([[dl1_dtheta1, dl1_dtheta2],
                          [dl2_dtheta1, dl2_dtheta2]])
    
    # 返回前向雅可比矩阵（用于IK求解）
    return J_forward

def inverse_kinematics_with_jacobian(l1_target, l2_target, l3_target, R, D1, D2, D3, D4, L, S, current_theta1=0.0, current_theta2=0.0, max_iterations=50, tolerance=1e-3):
    """
    使用雅可比矩阵迭代求解逆运动学（θ1,θ2用雅可比矩阵，θ3单独求解）
    
    参数:
    l1_target, l2_target, l3_target: 目标连杆长度
    R, D1, D2, D3, D4, L, S: 几何参数
    max_iterations: 最大迭代次数
    tolerance: 收敛容差
    
    返回:
    theta1, theta2, theta3: 关节角度（弧度）
    """
    # 首先单独求解θ3（根据论文公式）
    def solve_theta3(l3_target, R, D3, D4):
        """根据l3求解θ3"""
        # 使用数值方法求解：l3² = H3² + V3²
        def equation_theta3(theta3):
            H3 = D3 + R * np.sin(theta3) + D4 * np.cos(theta3)
            V3 = R * (1 - np.cos(theta3)) + D4 * np.sin(theta3)
            return l3_target**2 - (H3**2 + V3**2)
        
        # 尝试多个初始值
        for guess in [0.0, np.pi/6, -np.pi/6]:
            try:
                theta3 = fsolve(equation_theta3, guess)[0]
                if -np.pi/9 <= theta3 <= np.pi/9:
                    return theta3
            except:
                continue
        return None
    
    # 求解θ3
    theta3 = solve_theta3(l3_target, R, D3, D4)
    if theta3 is None:
        print("警告：无法求解θ3")
        return None, None, None
    
    # 使用雅可比矩阵迭代求解θ1, θ2
    # 使用当前角度作为初始值，提高收敛性
    theta = np.array([current_theta1, current_theta2])  # [theta1, theta2]
    l_target = np.array([l1_target, l2_target])
    
    # 添加自适应步长控制参数
    step_size = 0.5  # 增大初始步长因子，提高收敛速度
    prev_error_norm = float('inf')
    best_theta = theta.copy()  # 记录最佳解
    best_error = float('inf')
    
    for iteration in range(max_iterations):
        # 计算当前连杆长度
        theta1, theta2 = theta
        
        # 计算l1, l2（根据论文公式）
        A = R * (1 - np.cos(theta2)) + D2 * np.sin(theta1)
        H = D1 + R * np.sin(theta2) + D2 * np.cos(theta2)
        V = L / 2
        
        l1_current = np.sqrt(A**2 + (H * np.cos(theta1) - V * np.sin(theta1) - S)**2 + 
                            (H * np.sin(theta1) + V * np.cos(theta1) - V)**2)
        l2_current = np.sqrt(A**2 + (H * np.cos(theta1) + V * np.sin(theta1) - S)**2 + 
                            (H * np.sin(theta1) - V * np.cos(theta1) + V)**2)
        
        l_current = np.array([l1_current, l2_current])
        
        # 计算长度误差
        l_error = l_target - l_current
        current_error_norm = np.linalg.norm(l_error)
        
        # 记录最佳解
        if current_error_norm < best_error:
            best_error = current_error_norm
            best_theta = theta.copy()
        
        # 自适应步长控制
        if current_error_norm > prev_error_norm:
            step_size *= 0.7  # 如果误差增大，更大幅度减小步长
        else:
            step_size = min(step_size * 1.2, 1.0)  # 如果误差减小，适当增大步长
        
        prev_error_norm = current_error_norm
    
        current_joint_pos = np.array([theta1, theta2])  
        jacobian = compute_jacobian(current_joint_pos, R, D1, D2, L, S)
        
        # 使用雅可比矩阵更新关节角度
        try:
            # 使用伪逆避免奇异性
            jacobian_inv = np.linalg.pinv(jacobian)
            delta_theta = jacobian_inv @ l_error
            
            # 使用步长控制更新关节角度
            theta += step_size * delta_theta
            
            # 限制角度范围
            theta = np.clip(theta, -np.pi/9, np.pi/9)
            
        except np.linalg.LinAlgError:
            print(f"警告：第{iteration}次迭代中雅可比矩阵奇异")
            break
        # 检查收敛 - 使用更宽松的条件
        if current_error_norm < tolerance:
            # 检查解是否在合理范围内
            if (-np.pi/9 <= theta1 <= np.pi/9 and -np.pi/9 <= theta2 <= np.pi/9):
                return theta1, theta2, theta3
        
        # 如果步长太小，说明可能陷入局部最小值，尝试重新初始化
        if step_size < 0.01:
            theta = np.array([current_theta1, current_theta2])  # 重新初始化到当前角度
            step_size = 0.5
            prev_error_norm = float('inf')
    
    print(f"警告：迭代{max_iterations}次后未收敛，最终误差: {best_error:.6f}")
    # 返回最佳解而不是当前解
    if (-np.pi/9 <= best_theta[0] <= np.pi/9 and -np.pi/9 <= best_theta[1] <= np.pi/9):
        return best_theta[0], best_theta[1], theta3
    return None, None, None

# JETBOT_CONFIG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(usd_path=f"/home/baai/Projects/C/Dexhand/Dexhand_single/assets/hand/Assem_DexCo.usd"),
#     actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
# )

DOFBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/baai/Projects/C/Dexhand/Dexhand_single/assets/limit_1.57/Assem_DexCo_2/Assem_DexCo_2/Assem_DexCo_2.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "L1_joint": 0.0,
            "R1_joint": 0.0,
            "L2_pre_joint": 0.0,
            "R2_pre_joint": 0.0,
            "L3_pre_joint": 0.0,
            "R3_pre_joint": 0.0,

        },
        pos=(0.25, -0.25, 2.0),
    ),
    actuators={
        "L1_act": ImplicitActuatorCfg(
            joint_names_expr=["L1_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=500.0,
            damping=500.0,
        ),
        "R1_act": ImplicitActuatorCfg(
            joint_names_expr=["R1_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=500.0,
            damping=500.0,
        ),
        "L2_act": ImplicitActuatorCfg(
            joint_names_expr=["L2_pre_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=500.0,
            damping=500.0,
        ),
        "R2_act": ImplicitActuatorCfg(
            joint_names_expr=["R2_pre_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=500.0,
            damping=500.0,
        ),
        "L3_act": ImplicitActuatorCfg(
            joint_names_expr=["L3_pre_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=500.0,
            damping=500.0,
        ),
        "R3_act": ImplicitActuatorCfg(
            joint_names_expr=["R3_pre_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=500.0,
            damping=500.0,
        ),
    },
)


class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    # Jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")
    Dofbot = DOFBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Dofbot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # 初始化 current_action 为默认关节位置
    current_action = scene["Dofbot"].data.default_joint_pos.clone()
    # from pdb import set_trace
    # set_trace()
    
    # 逆运动学参数 - 根据论文中的参数范围调整
    R = 0.02    # 径向距离 (20mm)
    D1 = 0.015  # 偏移量1 (15mm) 
    D2 = 0.01   # 偏移量2 (10mm)
    D3 = 0.012  # 偏移量3 (12mm)
    D4 = 0.008  # 偏移量4 (8mm)
    L = 0.04    # 执行器端点间距离 (40mm)
    S = 0.005   # 偏移参数 (5mm)

    # 自动分析 l3 可达范围
    thetas = np.linspace(-np.pi/9, np.pi/9, 100)
    l3_vals = []
    for t in thetas:
        H3 = D3 + R * np.sin(t) + D4 * np.cos(t)
        V3 = R * (1 - np.cos(t)) + D4 * np.sin(t)
        l3 = np.sqrt(H3**2 + V3**2)
        l3_vals.append(l3)
    l3_min, l3_max = min(l3_vals), max(l3_vals)
    print(f"l3 可达范围: {l3_min:.4f} ~ {l3_max:.4f}")

    compliance_params = {
    'd0': 0.014,   # 初始内径 (m)
    'D0': 0.022,  # 初始外径 (m)
    'h0': 0.1,    # 初始长度 (m)
    't': 0.0002,   # 壁厚 (m)
    'E': 8.7e7      # 弹性模量 (Pa)
        }   
    Fh = 10.0
    # 柔顺性参数
    compliance_stiffness = 100.0  # 柔顺性刚度 (N⋅m/rad)
    enable_compliance = True      # 是否启用柔顺性
    
    # 外力参数（模拟外部力输入）
    external_force = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # [Fx, Fy, Fz, Mx, My, Mz]
    force_magnitude = 10.0  # 外力大小 (N)
    
    # 角度变化模式选择
    angle_mode = 4  # 1: 正弦余弦模式, 2: 圆形轨迹模式, 3: 线性扫描模式

    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            count = 0
            root_dofbot_state = scene["Dofbot"].data.default_root_state.clone()
            root_dofbot_state[:, :3] += scene.env_origins
            scene["Dofbot"].write_root_pose_to_sim(root_dofbot_state[:, :7])
            scene["Dofbot"].write_root_velocity_to_sim(root_dofbot_state[:, 7:])
            joint_pos, joint_vel = (
                scene["Dofbot"].data.default_joint_pos.clone(),
                scene["Dofbot"].data.default_joint_vel.clone(),
            )
            scene["Dofbot"].write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            print("[INFO]: Resetting Jetbot and Dofbot state...")
            # 重置 current_action
            current_action = scene["Dofbot"].data.default_joint_pos.clone()

        # 根据l1, l2, l3目标值求解theta1, theta2, theta3
        l1_idx = 0  # L1_joint的下标
        l2_idx = 2  # L2_pre_joint的下标
        l3_idx = 4  # L3_pre_joint的下标
        
        # 让l1, l2, l3随时间变化（根据论文中的实际范围调整）
        if angle_mode == 1:
            # 模式1：正弦余弦变化
            l1_target = 0.015 + 0.005 * np.sin(count * 0.01)  # 15mm ± 5mm
            l2_target = 0.020 + 0.005 * np.cos(count * 0.01)  # 20mm ± 5mm
            l3_target = (l3_min + l3_max) / 2 + (l3_max - l3_min) / 2 * np.sin(count * 0.015)
        elif angle_mode == 2:
            # 模式2：圆形轨迹
            angle = count * 0.02  # 更快的角度变化
            radius = 0.003  # 圆形轨迹半径
            l1_target = 0.015 + radius * np.cos(angle)
            l2_target = 0.020 + radius * np.sin(angle)
            l3_target = (l3_min + l3_max) / 2 + radius * np.sin(angle * 2)  # 双倍频率
        elif angle_mode == 3:
            # 模式3：线性扫描模式 - 使用更保守的范围
            scan_period = 400  # 增加扫描周期，使变化更缓慢
            phase = (count % scan_period) / scan_period  # 0到1之间的相位
            l1_target = 0.013 + 0.004 * phase  # 13mm到17mm线性变化（更保守）
            l2_target = 0.019 + 0.004 * (1 - phase)  # 19mm到23mm反向变化（更保守）
            l3_target = l3_min + (l3_max - l3_min) * 0.5 + (l3_max - l3_min) * 0.2 * np.sin(phase * 2 * np.pi)  # l3在中间范围附近变化，减小振幅
        elif angle_mode == 4:
            N = 50      # 每个自由度变化的步数
            PAUSE = 20  # 每 sweep 完一个自由度后静止的步数
            l_min, l_max = 0.013, 0.026
            l_mid = (l_min + l_max) / 2

            total_dim = 3  # sweep l1/l2/l3
            phase_len = N + PAUSE  # 每个自由度 sweep+静止的总步数
            total_cycle = total_dim * phase_len

            phase = (count // phase_len) % total_dim
            phase_step = count % phase_len

            # 默认都用中值
            l1_target = l_mid
            l2_target = l_mid
            l3_target = (l3_min + l3_max) / 2

            if phase_step < N:
                # sweep 阶段
                t = phase_step / (N - 1)
                value = l_min + (l_max - l_min) * t
                if phase == 0:
                    l1_target = value
                elif phase == 1:
                    l2_target = value
                elif phase == 2:
                    l3_target = l3_min + (l3_max - l3_min) * t
            # else:  # 静止阶段，全部保持中值
    #     l1_target, l2_target, l3_target = l_mid, l_mid, (l3_min + l3_max) / 2
        else:
            # 默认模式：静止状态
            l1_target = 0.015
            l2_target = 0.020
            l3_target = (l3_min + l3_max) / 2
        
        # 获取当前关节角度作为初始值
        current_joint_positions = scene["Dofbot"].data.joint_pos[0].cpu().numpy()
        current_theta1 = current_joint_positions[l1_idx]  # L1_joint的当前角度
        current_theta2 = current_joint_positions[l2_idx]  # L2_pre_joint的当前角度
        
        # 使用雅可比矩阵迭代求解逆运动学，传入当前角度作为初始值
        theta1, theta2, theta3 = inverse_kinematics_with_jacobian(
            l1_target, l2_target, l3_target, R, D1, D2, D3, D4, L, S,
            current_theta1, current_theta2
        )
        
        target_action = scene["Dofbot"].data.default_joint_pos.clone()
        
        if theta1 is not None and theta2 is not None and theta3 is not None:
            # 设置L1_joint、L2_pre_joint和L3_pre_joint的角度
            # 将numpy类型转换为torch tensor类型
            target_action[:, l1_idx] = torch.tensor(theta1, device=target_action.device, dtype=target_action.dtype)
            target_action[:, l2_idx] = torch.tensor(theta2, device=target_action.device, dtype=target_action.dtype)
            target_action[:, l3_idx] = torch.tensor(theta3, device=target_action.device, dtype=target_action.dtype)
            
            # 如果启用柔顺性，计算并应用柔顺性扭矩
            if enable_compliance:
                # 获取当前关节位置
                current_joint_pos = scene["Dofbot"].data.joint_pos[0][:2].cpu().numpy()
                
                # 模拟外力输入（可以根据需要调整）
                # 例如：在特定时间施加外力
                # if count > 100 and count < 200:  # 在100-200步之间施加外力
                Fx, Fy = 0, 0
                external_force = np.array([Fx, Fy])
                    # external_force = np.array([force_magnitude, 0.0, 0.0, 0.0, 0.0, 0.0])  # X方向力
                # else:
                    # external_force = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    # external_force = np.array([0, 0])
                # from pdb import set_trace
                # set_trace()
                current_joint_pos = scene["Dofbot"].data.joint_pos[0][:2].cpu().numpy()
                jacobian = compute_jacobian(current_joint_pos, R, D1, D2, L, S)
                h, Ch, Cl = compliance_cl(compliance_params, Fh)
                Cq = Cl * np.linalg.inv(jacobian.T @ jacobian)
                external_force = np.array([Fx, Fy])
                delta_q = Cq @ (jacobian.T @ external_force)
                tau = jacobian.T @ external_force
                # # 应用柔顺性扭矩（通过调整目标位置）
                # compliance_adjustment = compliance_torque * 0.001  # 缩放因子
                delta_q_tensor = torch.from_numpy(delta_q).to(target_action.device, dtype=target_action.dtype)
                target_action[0, :2] += delta_q_tensor
        else:
            # 如果求解失败，保持当前角度不变
            print(f"逆运动学求解失败，l1={l1_target:.3f}, l2={l2_target:.3f}, l3={l3_target:.3f}")

        scene["Dofbot"].set_joint_position_target(target_action)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        if count % 50 == 0:
            joint_names = scene["Dofbot"].data.joint_names
            joint_positions = scene["Dofbot"].data.joint_pos  # 这是一个tensor或numpy数组
            for name, pos in zip(joint_names, joint_positions[0]):
                print(f"{name}: {pos:.4f}")
            # 打印当前目标l1, l2, l3值
            print(f"角度模式: {angle_mode}")
            print(f"目标l1: {l1_target:.3f}, 目标l2: {l2_target:.3f}, 目标l3: {l3_target:.3f}")
            print(f"当前角度: theta1={current_theta1:.4f} rad ({np.degrees(current_theta1):.2f}°), "
                  f"theta2={current_theta2:.4f} rad ({np.degrees(current_theta2):.2f}°)")
            if theta1 is not None:
                print(f"求解角度: theta1={theta1:.4f} rad ({np.degrees(theta1):.2f}°), "
                      f"theta2={theta2:.4f} rad ({np.degrees(theta2):.2f}°), "
                      f"theta3={theta3:.4f} rad ({np.degrees(theta3):.2f}°)")
                if enable_compliance:
                    current_joint_pos = scene["Dofbot"].data.joint_pos[0]
                    # 计算当前外力
                    if count > 100 and count < 200:  # 在100-200步之间施加外力
                        Fx, Fy = 10, 0
                        external_force = np.array([Fx, Fy])
                        # external_force = np.array([force_magnitude, 0.0, 0.0, 0.0, 0.0, 0.0])  # X方向力
                    else:
                        # external_force = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                        external_force = np.array([0, 0])
                
                    current_joint_pos = scene["Dofbot"].data.joint_pos[0][:2].cpu().numpy()
                    jacobian = compute_jacobian(current_joint_pos, R, D1, D2, L, S)
                    h, Ch, Cl = compliance_cl(compliance_params, Fh)
                    Cq = Cl * np.linalg.inv(jacobian.T @ jacobian)
                    external_force = np.array([Fx, Fy])
                    delta_q = Cq @ (jacobian.T @ external_force)
                    tau = jacobian.T @ external_force
                    print(f"外力: [{external_force[0]:.2f}, {external_force[1]:.2f}] N")
                    print(f"基于外力的柔顺性扭矩: [{tau[0]:.4f}, {tau[1]:.4f}] N⋅m")
            print("-" * 50)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()