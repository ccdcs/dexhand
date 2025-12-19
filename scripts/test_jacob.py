# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

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

def compute_jacobian(joint_positions, R, D1, D2, D3, D4, L, S):
    """
    计算雅可比矩阵（根据论文公式）
    
    参数:
    joint_positions: 关节位置 [theta1, theta2, theta3, ...]
    R, D1, D2, L, S: 几何参数
    
    返回:
    jacobian: 雅可比矩阵 J =  [∂l1/∂θ1  ∂l2/θ1  0]
                            [∂l1/∂θ2  ∂l2/∂θ2  0]
                            [0       0       ∂l3/∂θ3]
    """
    # 确保输入是numpy数组
    if hasattr(joint_positions, 'cpu'):
        joint_positions = joint_positions.cpu().numpy()
    
    theta1, theta2, theta3 = joint_positions
    
    # 计算中间变量
    A = R * (1 - np.cos(theta2)) + D2 * np.sin(theta1)
    H = D1 + R * np.sin(theta2) + D2 * np.cos(theta2)
    V = L / 2
    
    # 计算 l1 和 l2 的表达式（从论文公式(1)）
    l1_squared = A**2 + (H * np.cos(theta1) - V * np.sin(theta1) - S)**2 + (H * np.sin(theta1) + V * np.cos(theta1) - V)**2
    l2_squared = A**2 + (H * np.cos(theta1) + V * np.sin(theta1) - S)**2 + (H * np.sin(theta1) - V * np.cos(theta1) + V)**2
    
    # 计算H3和V3（根据论文公式(5)）
    H3 = D3 + R * np.sin(theta3) + D4 * np.cos(theta3)
    V3 = R * (1 - np.cos(theta3)) + D4 * np.sin(theta3)
    l3_squared = H3**2 + V3**2

    l1 = np.sqrt(l1_squared)
    l2 = np.sqrt(l2_squared)
    l3 = np.sqrt(l3_squared)
    
    
    # ∂l1/∂θ1（根据论文公式(1)）
    # l1² = A² + (H cos θ1 - V sin θ1 - S)² + (H sin θ1 + V cos θ1 - V)²
    # ∂(l1²)/∂θ1 = 2A * ∂A/∂θ1 + 2(H cos θ1 - V sin θ1 - S) * ∂(H cos θ1 - V sin θ1 - S)/∂θ1 + ...
    # 其中 ∂A/∂θ1 = D2 * cos(θ1)
    dA_dtheta1 = D2 * np.cos(theta1)
    dA_dtheta2 = R * np.sin(theta2)
    dH_dtheta2 = R * np.cos(theta2) - D2 * np.sin(theta2)
    
    dl1_dtheta1 = (A * dA_dtheta1 + \
                   (H * np.cos(theta1) - V * np.sin(theta1) - S) * (-H * np.sin(theta1) - V * np.cos(theta1)) + \
                   (H * np.sin(theta1) + V * np.cos(theta1) - V) * (H * np.cos(theta1) - V * np.sin(theta1))) / l1
    
    # ∂l1/∂θ2
    dl1_dtheta2 = (A * dA_dtheta2 + \
                   (H * np.cos(theta1) - V * np.sin(theta1) - S) * dH_dtheta2 * np.cos(theta1) + \
                   (H * np.sin(theta1) + V * np.cos(theta1) - V) * dH_dtheta2 * np.sin(theta1)) / l1
    dl1_dtheta3 = 0  

    # ∂l2/∂θ1
    dl2_dtheta1 = (A * dA_dtheta1 + \
                   (H * np.cos(theta1) + V * np.sin(theta1) - S) * (-H * np.sin(theta1) + V * np.cos(theta1)) + \
                   (H * np.sin(theta1) - V * np.cos(theta1) + V) * (H * np.cos(theta1) + V * np.sin(theta1))) / l2
    
    # ∂l2/∂θ2
    dl2_dtheta2 = (A * dA_dtheta2 + \
                   (H * np.cos(theta1) + V * np.sin(theta1) - S) * dH_dtheta2 * np.cos(theta1) + \
                   (H * np.sin(theta1) - V * np.cos(theta1) + V) * dH_dtheta2 * np.sin(theta1)) / l2
    dl2_dtheta3 = 0

    # ∂l3/∂θ3（根据论文公式(5)）
    # l3² = H3² + V3²
    # ∂(l3²)/∂θ3 = 2H3 * ∂H3/∂θ3 + 2V3 * ∂V3/∂θ3
    # ∂H3/∂θ3 = R cos θ3 - D4 sin θ3
    # ∂V3/∂θ3 = R sin θ3 + D4 cos θ3
    dH3_dtheta3 = R * np.cos(theta3) - D4 * np.sin(theta3)
    dV3_dtheta3 = R * np.sin(theta3) + D4 * np.cos(theta3)
    dl3_dtheta3 = (H3 * dH3_dtheta3 + V3 * dV3_dtheta3) / l3
    
    dl3_dtheta1 = 0  
    dl3_dtheta2 = 0

    J_forward = np.array([[dl1_dtheta1, dl1_dtheta2, dl1_dtheta3],
                        [dl2_dtheta1, dl2_dtheta2, dl2_dtheta3],
                        [dl3_dtheta1, dl3_dtheta2, dl3_dtheta3]])
    
    # 返回逆雅可比矩阵（从关节角度到线缆长度）
    return J_forward

def compute_hydraulic_compliance_stiffness(material_params):
    """
    根据软管材料参数计算液压柔顺性刚度
    
    参数:
    material_params: 材料参数字典
        - d0: 初始内径 (m)
        - D0: 初始外径 (m)
        - h0: 初始长度 (m)
        - t: 壁厚 (m)
        - E: 弹性模量 (Pa)
    
    返回:
    Cl: 有效线性柔顺性 (m/(N·m²))
    """
    d0 = material_params['d0']
    D0 = material_params['D0']
    h0 = material_params['h0']
    t = material_params['t']
    E = material_params['E']
    
    # 计算初始体积
    V0 = (np.pi * h0 / 3) * ((D0 / 2)**2 + (d0 / 2)**2 + (D0 * d0) / 4)
    
    # 计算有效截面积
    A = np.pi * d0**2 / 4
    
    # 计算柔顺性Ch（使用有限差分法）
    delta = 1e-4  # 小的力增量
    Fh = 10.0  # 基准力
    
    # 计算变形后的直径
    P = -4 * Fh / (np.pi * d0**2)
    d = d0 + (P * d0**2) / (2 * E * t)
    D = D0 + (P * D0**2) / (2 * E * t)
    
    # 求解变形后长度
    def equation(h):
        V = (np.pi * h / 3) * ((D / 2)**2 + (d / 2)**2 + (D * d) / 4)
        return V - V0
    
    h1 = fsolve(equation, h0)[0]
    
    # 计算力增量后的变形
    P2 = -4 * (Fh + delta) / (np.pi * d0**2)
    d2 = d0 + (P2 * d0**2) / (2 * E * t)
    D2 = D0 + (P2 * D0**2) / (2 * E * t)
    
    def equation2(h):
        V = (np.pi * h / 3) * ((D2 / 2)**2 + (d2 / 2)**2 + (D2 * d2) / 4)
        return V - V0
    
    h2 = fsolve(equation2, h1)[0]
    
    # 计算柔顺性Ch
    Ch = (h2 - h1) / delta
    
    # 计算有效线性柔顺性Cl
    Cl = Ch / A
    
    return Cl


def compute_compliance_torque_with_length_difference(l_current, l_target, joint_positions, material_params):
    """
    基于长度差异的柔顺性扭矩计算（根据论文公式）
    
    参数:
    l_current: 当前连杆长度 [l1, l2, l3]
    l_target: 目标连杆长度 [l1_target, l2_target, l3_target]
    joint_positions: 当前关节位置 [theta1, theta2, theta3]
    material_params: 材料参数
    
    返回:
    compliance_torque: 柔顺性扭矩
    """
    # 确保输入是numpy数组
    if hasattr(joint_positions, 'cpu'):
        joint_positions = joint_positions.cpu().numpy()
    
    # 计算长度差异
    delta_l = np.array(l_target) - np.array(l_current)
    
    # 计算雅可比矩阵
    jacobian = compute_jacobian(joint_positions[:3], R=15, D1=10.5, D2=5, L=27, S=3.5, D3=3.58, D4=3.5)
    
    # 计算液压柔顺性刚度
    Cl = compute_hydraulic_compliance_stiffness(material_params)
    
    # 根据论文公式：Δτ = (1/Cl) * J^T * Δl
    compliance_torque = np.dot(jacobian.T, delta_l)  
    
    # 应用柔顺性缩放
    compliance_torque = compliance_torque * (1.0 / Cl)
    
    return compliance_torque, Cl


def inverse_kinematics(l1, l2, l3, R, D1, D2, D3, D4, L, S):
    """
    根据连杆长度l1, l2, l3求解关节角度theta1, theta2, theta3
    
    参数:
    l1, l2, l3: 目标连杆长度
    R: 径向距离
    D1, D2, D3, D4: 偏移量
    L: 执行器端点间距离
    S: 偏移参数
    
    返回:
    theta1, theta2, theta3: 关节角度（弧度）
    """
    def equations(vars):
        theta1, theta2, theta3 = vars
        
        # 根据论文，A, H, V是theta2的函数
        A = R * (1 - np.cos(theta2)) + D2 * np.sin(theta1)
        H = D1 + R * np.sin(theta2) + D2 * np.cos(theta2)
        V = L / 2  # 修正：使用L/2，与正向运动学保持一致
        
        # 三个方程
        eq1 = l1**2 - (A**2 + (H * np.cos(theta1) - V * np.sin(theta1) - S)**2 + 
                       (H * np.sin(theta1) + V * np.cos(theta1) - V)**2)
        
        eq2 = l2**2 - (A**2 + (H * np.cos(theta1) + V * np.sin(theta1) - S)**2 + 
                       (H * np.sin(theta1) - V * np.cos(theta1) + V)**2)
        
        # 第三个方程 - 根据提供的正确公式
        H3 = D3 + R * np.sin(theta3) + D4 * np.cos(theta3)
        V3 = R * (1 - np.cos(theta3)) + D4 * np.sin(theta3)
        eq3 = l3**2 - (H3**2 + V3**2)
        
        return [eq1, eq2, eq3]
    
    # 扩展初始猜测值，提高求解成功率
    initial_guesses = [
        [0.0, 0.0, 0.0],
        [np.pi / 6, np.pi / 6, np.pi / 6],    # 30度
        [-np.pi / 6, -np.pi / 6, -np.pi / 6],  # -30度
        [np.pi / 6, -np.pi / 6, np.pi / 6],   # 30度, -30度, 30度
        [-np.pi / 6, np.pi / 6, -np.pi / 6],   # -30度, 30度, -30度
        [0.1, 0.1, 0.1],                       # 小角度
        [-0.1, -0.1, -0.1],                    # 负小角度
        [0.2, 0.2, 0.2],                       # 中等角度
        [-0.2, -0.2, -0.2],                    # 负中等角度
        [0.0, 0.2, 0.0],                       # 只有theta2
        [0.0, -0.2, 0.0],                      # 只有负theta2
        [0.1, 0.0, 0.1],                       # theta1和theta3
        [-0.1, 0.0, -0.1],                     # 负theta1和theta3
        [0.05, 0.05, 0.05],                    # 更小的角度
        [-0.05, -0.05, -0.05],                 # 负更小角度
        # 添加更多针对极端情况的猜测
        [0.3, 0.0, 0.3],                       # 大角度
        [-0.3, 0.0, -0.3],                     # 负大角度
        [0.0, 0.3, 0.0],                       # 只有大theta2
        [0.0, -0.3, 0.0],                      # 只有负大theta2
        [0.4, 0.4, 0.4],                       # 极大角度
        [-0.4, -0.4, -0.4],                    # 负极大角度
    ]
    
    best_solution = None
    best_error = float('inf')
    
    for guess in initial_guesses:
        try:
            # 使用更宽松的收敛条件
            result = fsolve(equations, guess, maxfev=3000, xtol=1e-6, full_output=True)
            theta1, theta2, theta3 = result[0]
            success = result[1]
            
            if not success:  # 如果求解失败，跳过
                continue
                
            # 计算方程误差
            eq_values = equations([theta1, theta2, theta3])
            error = np.sqrt(sum(eq**2 for eq in eq_values))
            
            # 放宽角度范围限制到[-35°, 35°]
            if (-np.pi / 5 <= theta1 <= np.pi / 5 and -np.pi / 5 <= theta2 <= np.pi / 5 and -np.pi / 5 <= theta3 <= np.pi / 5):
                if error < best_error:
                    best_solution = (theta1, theta2, theta3)
                    best_error = error
                    
        except Exception:
            continue
    
    # 进一步放宽误差阈值
    if best_solution is not None and best_error < 1e-3:
        return best_solution
    
    # 如果常规方法失败，尝试更激进的策略
    print("尝试激进求解策略...")
    
    # 使用更宽松的误差阈值和更大的角度范围
    for guess in initial_guesses:
        try:
            result = fsolve(equations, guess, maxfev=5000, xtol=1e-4, full_output=True)
            theta1, theta2, theta3 = result[0]
            success = result[1]
            
            if not success:
                continue
                
            eq_values = equations([theta1, theta2, theta3])
            error = np.sqrt(sum(eq**2 for eq in eq_values))
            
            # 更宽松的角度范围限制到[-45°, 45°]
            if (-np.pi / 4 <= theta1 <= np.pi / 4 and -np.pi / 4 <= theta2 <= np.pi / 4 and -np.pi / 4 <= theta3 <= np.pi / 4):
                if error < best_error:
                    best_solution = (theta1, theta2, theta3)
                    best_error = error
                    
        except Exception:
            continue
    
    # 最终放宽误差阈值
    if best_solution is not None and best_error < 1e-2:
        return best_solution
    
    # 如果所有初始猜测都失败，尝试使用更宽松的条件
    print(f"警告：无法找到有效解，最小误差: {best_error:.6f}")
    return None, None, None


DOFBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/limit_1.57/Assem_DexCo_2/Assem_DexCo_2/Assem_DexCo_2.usd",
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
        pos=(0.25, -0.25, 5.0),
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

    R = 15
    D1 = 10.5
    D2 = 5
    S = 3.5
    L = 27
    # R = 15
    D3 = 3.58
    D4 = 3.5

    enable_compliance = True      # 是否启用柔顺性
    
    # 材料参数（用于计算柔顺性刚度）
    material_params = {
        'd0': 0.014,    # 14mm 内径
        'D0': 0.022,    # 22mm 外径
        'h0': 0.023,    # 23mm 初始长度
        't': 0.0002,    # 0.2mm 壁厚
        'E': 87e6       # 87MPa 弹性模量
    }

    # 添加统计变量
    total_attempts = 0
    successful_solves = 0
    failed_solves = 0
    angle_errors = []  # 存储角度误差
    length_errors = []  # 存储连杆长度误差

    while simulation_app.is_running():
        # reset
        if count % 2000 == 0:
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

        # 根据l1, l2, l3目标值求解theta1, theta2, theta3
        l1_idx = 0  # L1_joint的下标
        l2_idx = 2  # L2_pre_joint的下标
        l3_idx = 4  # L3_pre_joint的下标
        
        # 从CSV文件读取实际的l1, l2, l3值（01-06所有文件）
        import pandas as pd
        import matplotlib.pyplot as plt
        
        if count == 0:  # 只在第一次读取，避免重复读取
            all_data = []
            for i in range(1, 7):
                df_temp = pd.read_csv(f'actuation2configuration/aruco_joint_angles_0{i}.csv')
                all_data.append(df_temp)
            df = pd.concat(all_data, ignore_index=True)
            # 保存为全局变量
            run_simulator.df = df
            
            # 初始化存储数组
            run_simulator.theta1_orig_list = []
            run_simulator.theta2_orig_list = []
            run_simulator.theta3_orig_list = []
            run_simulator.theta1_solved_list = []
            run_simulator.theta2_solved_list = []
            run_simulator.theta3_solved_list = []
            run_simulator.row_indices = []
        
        # 根据count选择对应的行（循环使用数据）
        row_idx = count % len(df)
        l1_target = run_simulator.df.iloc[row_idx]['l1']  # 转换为米
        l2_target = run_simulator.df.iloc[row_idx]['l2']  # 转换为米
        l3_target = run_simulator.df.iloc[row_idx]['l3']  # 转换为米

        theta1_orig = run_simulator.df.iloc[row_idx]['joint_angle_x']
        theta2_orig = run_simulator.df.iloc[row_idx]['joint_angle_y'] 
        theta3_orig = run_simulator.df.iloc[row_idx]['joint_angle_z']
    
        # 求解逆运动学
        theta1, theta2, theta3 = inverse_kinematics(l1_target, l2_target, l3_target, R, D1, D2, D3, D4, L, S)
  
        target_action = scene["Dofbot"].data.default_joint_pos.clone()
        # 更新统计
        total_attempts += 1

        if theta1 is not None:
            successful_solves += 1
            
            # 存储数据用于画图
            run_simulator.theta1_orig_list.append(theta1_orig)
            run_simulator.theta2_orig_list.append(theta2_orig)
            run_simulator.theta3_orig_list.append(theta3_orig)
            run_simulator.theta1_solved_list.append(theta1)
            run_simulator.theta2_solved_list.append(theta2)
            run_simulator.theta3_solved_list.append(theta3)
            run_simulator.row_indices.append(row_idx)
            
            # 计算角度误差
            error1 = abs(theta1_orig - theta1)
            error2 = abs(theta2_orig - theta2)
            error3 = abs(theta3_orig - theta3)
            max_angle_error = max(error1, error2, error3)
            angle_errors.append(max_angle_error)
            
            # 计算连杆长度误差（验证求解结果）
            def calc_l1_l2(theta1, theta2, theta3):
                A = R * (1 - np.cos(theta2)) + D2 * np.sin(theta1)
                H = D1 + R * np.sin(theta2) + D2 * np.cos(theta2)
                V = L / 2
                
                l1_calc = np.sqrt(A**2 + (H * np.cos(theta1) - V * np.sin(theta1) - S)**2 + 
                                  (H * np.sin(theta1) + V * np.cos(theta1) - V)**2)
                l2_calc = np.sqrt(A**2 + (H * np.cos(theta1) + V * np.sin(theta1) - S)**2 + 
                                  (H * np.sin(theta1) - V * np.cos(theta1) + V)**2)
                
                H3 = D3 + R * np.sin(theta3) + D4 * np.cos(theta3)
                V3 = R * (1 - np.cos(theta3)) + D4 * np.sin(theta3)
                l3_calc = np.sqrt(H3**2 + V3**2)
                
                return l1_calc, l2_calc, l3_calc
            
            l1_calc, l2_calc, l3_calc = calc_l1_l2(theta1, theta2, theta3)
            length_error1 = abs(l1_target - l1_calc)
            length_error2 = abs(l2_target - l2_calc)
            length_error3 = abs(l3_target - l3_calc)
            max_length_error = max(length_error1, length_error2, length_error3)
            length_errors.append(max_length_error)
            
            # 计算成功率
            success_rate = (successful_solves / total_attempts) * 100
            
            # 计算平均误差
            avg_angle_error = np.mean(angle_errors) if angle_errors else 0
            avg_length_error = np.mean(length_errors) if length_errors else 0

            print(f'count:{count} | total:{len(run_simulator.df)}')
            print(f"{row_idx:<4} θ1:{theta1_orig:8.4f} θ2:{theta2_orig:8.4f} θ3:{theta3_orig:8.4f} | "
                  f"θ1:{theta1:8.4f} θ2:{theta2:8.4f} θ3:{theta3:8.4f} | "
                  f"e1:{error1:6.4f} e2:{error2:6.4f} e3:{error3:6.4f}")
            print(f"✅ 成功 | 成功率: {success_rate:.2f}% | 平均角度误差: {avg_angle_error:.6f} | 平均长度误差: {avg_length_error:.6f}")
        else:
            failed_solves += 1
            success_rate = (successful_solves / total_attempts) * 100
            print(f"{row_idx:<4} 求解失败")
            print(f"❌ 失败 | 成功率: {success_rate:.2f}% | 总尝试: {total_attempts} | 成功: {successful_solves} | 失败: {failed_solves}")
        
        target_action = scene["Dofbot"].data.default_joint_pos.clone()

        if theta1 is not None and theta2 is not None and theta3 is not None:
            # 设置L1_joint、L2_pre_joint和L3_pre_joint的角度
            target_action[:, l1_idx] = theta1
            target_action[:, l2_idx] = theta2
            target_action[:, l3_idx] = theta3
            
            # 如果启用柔顺性，计算并应用柔顺性扭矩
            if enable_compliance:
                # 获取当前关节位置
                current_joint_pos = scene["Dofbot"].data.joint_pos[0]
                
                # 计算当前连杆长度（基于当前关节角度）
                def calc_current_l_lengths(theta1, theta2, theta3):
                    A = R * (1 - np.cos(theta2)) + D2 * np.sin(theta1)
                    H = D1 + R * np.sin(theta2) + D2 * np.cos(theta2)
                    V = L / 2
                    
                    l1_current = np.sqrt(A**2 + (H * np.cos(theta1) - V * np.sin(theta1) - S)**2 + 
                                        (H * np.sin(theta1) + V * np.cos(theta1) - V)**2)
                    l2_current = np.sqrt(A**2 + (H * np.cos(theta1) + V * np.sin(theta1) - S)**2 + 
                                        (H * np.sin(theta1) - V * np.cos(theta1) + V)**2)
                    
                    H3 = D3 + R * np.sin(theta3) + D4 * np.cos(theta3)
                    V3 = R * (1 - np.cos(theta3)) + D4 * np.sin(theta3)
                    l3_current = np.sqrt(H3**2 + V3**2)
                    
                    return [l1_current, l2_current, l3_current]
                
                # 获取当前连杆长度（将CUDA张量转换为numpy数组）
                current_joint_pos_cpu = current_joint_pos.cpu().numpy()
                l_current = calc_current_l_lengths(current_joint_pos_cpu[0], current_joint_pos_cpu[2], current_joint_pos_cpu[4])
                l_target = [l1_target, l2_target, l3_target]
                
                # 使用基于长度差异的柔顺性计算
                compliance_torque, Cl = compute_compliance_torque_with_length_difference(
                    l_current, l_target, current_joint_pos, material_params
                )

                print(f"柔顺性扭矩计算成功 扭矩：{compliance_torque} N⋅m | 液压柔顺性刚度 Cl: {Cl:.6f} m/(N·m²)")

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
            print(f"目标l1: {l1_target:.3f}, 目标l2: {l2_target:.3f}, 目标l3: {l3_target:.3f}")
            if theta1 is not None:
                print(f"求解角度: theta1={theta1:.4f} rad ({np.degrees(theta1):.2f}°), "
                      f"theta2={theta2:.4f} rad ({np.degrees(theta2):.2f}°), "
                      f"theta3={theta3:.4f} rad ({np.degrees(theta3):.2f}°)")
            
            # 打印详细统计信息
            print("=" * 60)
            print(f"📊 逆运动学算法统计:")
            print(f"   总尝试次数: {total_attempts}")
            print(f"   成功次数: {successful_solves}")
            print(f"   失败次数: {failed_solves}")
            print(f"   成功率: {success_rate:.2f}%")
            if angle_errors:
                print(f"   平均角度误差: {avg_angle_error:.6f} rad ({np.degrees(avg_angle_error):.4f}°)")
                print(f"   最大角度误差: {max(angle_errors):.6f} rad ({np.degrees(max(angle_errors)):.4f}°)")
            if length_errors:
                print(f"   平均长度误差: {avg_length_error:.6f} mm")
                print(f"   最大长度误差: {max(length_errors):.6f} mm")
            print("=" * 60)
        
        # 在仿真结束时画图
        if count >= len(run_simulator.df):  # 运行完所有数据点后停止并画图
            print("仿真结束，开始画图...")
            
            # 创建子图
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # θ1 对比图
            axes[0].plot(run_simulator.row_indices, run_simulator.theta1_orig_list, 'b-', label='原始θ1', linewidth=2)
            axes[0].plot(run_simulator.row_indices, run_simulator.theta1_solved_list, 'r--', label='求解θ1', linewidth=2)
            axes[0].set_xlabel('数据点序号')
            axes[0].set_ylabel('θ1 (弧度)')
            axes[0].set_title('θ1 角度对比')
            axes[0].legend()
            axes[0].grid(True)
            
            # θ2 对比图
            axes[1].plot(run_simulator.row_indices, run_simulator.theta2_orig_list, 'b-', label='原始θ2', linewidth=2)
            axes[1].plot(run_simulator.row_indices, run_simulator.theta2_solved_list, 'r--', label='求解θ2', linewidth=2)
            axes[1].set_xlabel('数据点序号')
            axes[1].set_ylabel('θ2 (弧度)')
            axes[1].set_title('θ2 角度对比')
            axes[1].legend()
            axes[1].grid(True)
            
            # θ3 对比图
            axes[2].plot(run_simulator.row_indices, run_simulator.theta3_orig_list, 'b-', label='原始θ3', linewidth=2)
            axes[2].plot(run_simulator.row_indices, run_simulator.theta3_solved_list, 'r--', label='求解θ3', linewidth=2)
            axes[2].set_xlabel('数据点序号')
            axes[2].set_ylabel('θ3 (弧度)')
            axes[2].set_title('θ3 角度对比')
            axes[2].legend()
            axes[2].grid(True)
            
            plt.tight_layout()
            plt.savefig('angle_comparison_jacob.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("角度对比图已保存为 angle_comparison_jacob.png")
            simulation_app.close()
            # break  # 退出仿真循环


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
